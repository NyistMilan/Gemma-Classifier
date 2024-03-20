import click
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data_model.shroom_dataset import ShroomDataset
from data_model.scores import Scores
from model.gemma_classifier import GemmaClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_dataset(dataset, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for data_point in dataset:
        tokenized_concatenated = tokenizer(data_point['src'], data_point['hyp'],
                                           max_length=1024,
                                           truncation=True, padding='max_length', return_tensors="pt")

        data_point['input_ids'] = tokenized_concatenated['input_ids']
        data_point['attention_mask'] = tokenized_concatenated['attention_mask']

def collate(batch):
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def compute_metrics(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1

def train(model, train_loader, loss_fn, optimizer):
    model.train()

    train_loss = 0
    train_correct = 0

    for batch in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad()
        output, prediction = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
        )
        labels = batch['labels'].float().to(device)

        loss = loss_fn(output, labels)
        train_loss += loss
        train_correct += torch.eq(prediction, labels).sum().item()

        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader)
    return train_correct, train_loss

def eval(model, val_loader, num_labels, loss_fn):
    model.eval()

    dev_correct = 0
    dev_correct_by_labels = torch.zeros(num_labels, dtype=torch.int32, device=device)
    dev_labels_total = 0

    with torch.no_grad():
        dev_loss = 0
        true_positives = torch.zeros(num_labels)
        false_positives = torch.zeros(num_labels)
        false_negatives = torch.zeros(num_labels)
        for batch in val_loader:
            output, prediction = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device)
            )
            target = batch['labels'].float().to(device)
            
            dev_loss += loss_fn(output, target)
            dev_correct += torch.eq(prediction, target).sum().item()

            true_positives += (prediction * target).sum(dim=0).cpu()
            false_positives += (prediction * (1 - target)).sum(dim=0).cpu()
            false_negatives += ((1 - prediction) * target).sum(dim=0).cpu()

            dev_correct_by_labels += torch.eq(prediction, target).sum(dim=0)
            dev_labels_total += target.size(0)

    precision, recall, f1 = compute_metrics(true_positives, false_positives, false_negatives)
    dev_loss = dev_loss / len(val_loader)

    return dev_correct, dev_loss, dev_correct_by_labels, dev_labels_total, precision, recall, f1

@click.command()
@click.option('--model_name', default='google/gemma-2b')
@click.option('--path', default='')
@click.option('--num_labels', default=2)
@click.option('--epochs', default=10)
@click.option('--batch_size', default=4)
@click.option('--lr', default=1e-5)
@click.option('--train_flag', default=True)
@click.option('--patience', default=10)
@click.option('--output_path', default='')
def main(model_name, path, num_labels, epochs, batch_size, lr, train_flag, patience, output_path):
    dataset = ShroomDataset(path)
    tokenize_dataset(dataset, model_name)
    train_dataset, val_dataset = dataset.get_split()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = GemmaClassifier(model_name=model_name,
                                 num_labels=num_labels,
                                 train=train_flag).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_scores = Scores(loss=1000)
    curr_patience = 0
    early_stop = False

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_correct, train_loss = train(model, train_loader, loss_fn, optimizer)
        dev_correct, dev_loss, dev_correct_by_labels, dev_labels_total, precision, recall, f1 = eval(model,
                                                                                                     val_loader,
                                                                                                     num_labels, loss_fn)

        print(f'\tTrain_loss: {train_loss}')
        print(f'\tValid_loss: {dev_loss}')
        print(f'\tTrain_acc: {train_correct / (len(train_dataset) * num_labels)}')
        print(f'\tValid_acc: {dev_correct / (len(val_dataset) * num_labels)}')
        print(f'\tValid_acc_by_labels: {dev_correct_by_labels / dev_labels_total}')
        print(f'\tValid_precision_by_labels: {precision}')
        print(f'\tValid_recall_by_labels: {recall}')
        print(f'\tValid_f1_by_labels: {f1}')

        if best_scores.loss - dev_loss > 0.001:
            best_scores = Scores(loss=dev_loss,
                                 accuracy=dev_correct / (len(val_dataset) * num_labels),
                                 accuracy_by_labels=(dev_correct_by_labels / dev_labels_total).tolist(),
                                 precision=precision.tolist(),
                                 recall=recall.tolist(),
                                 f1=f1.tolist())
            curr_patience = 0
            torch.save(model.state_dict(), f"{output_path}/best_model_state.pth")
        else:
            curr_patience += 1
            if curr_patience == patience:
                early_stop = True

        if early_stop:
            torch.save(model.state_dict(), f"{output_path}/best_model_state.pth")
            with open(f'results.tsv', 'a') as f:
                f.write(f"{model_name}\t"
                        f"{train}\t"
                        f"{num_labels}\t"
                        f"{batch_size}\t"
                        f"{lr}\t"
                        f"{epoch + 1}\t"
                        f"{best_scores.loss}\t"
                        f"{best_scores.accuracy}\t"
                        f"{best_scores.accuracy_by_labels}\t"
                        f"{best_scores.precision}\t"
                        f"{best_scores.recall}\t"
                        f"{best_scores.f1}\n")
            break

if __name__ == "__main__":
    main()