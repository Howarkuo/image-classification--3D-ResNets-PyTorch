import torch
from torch.utils.tensorboard import SummaryWriter
from config import EPOCHS, DEVICE

def train_model(model, train_loader, val_loader, loss_function, optimizer):
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()

    for epoch in range(EPOCHS):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(DEVICE), batch_data[1].to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_len = len(train_loader.dataset) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            num_correct = 0.0
            metric_count = 0

            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(DEVICE), val_data[1].to(DEVICE)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

            metric = num_correct / metric_count
            metric_values.append(metric)

            if metric >= best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d.pth")
                print("Saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f}")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    
    return epoch_loss_values, metric_values
