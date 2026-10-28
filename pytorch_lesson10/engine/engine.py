import torch

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_grad_norm=1
):

    """
    Train the model for one epoch
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        # clear previous gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)

        # calculate loss
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm
        )

        # optimise model parameters
        optimizer.step()

        # track loss
        running_loss += loss.item()

        # predictions
        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy

# validation function
@torch.no_grad() # This prevents PyTorch from building a computation graph during validation.
                # That saves memory and computation.

def validate(
    model,
    dataloader,
    criterion,
    device
):

    """
    Validate the model for one epoch
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy

# check checkpoint function
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    best_val_loss,
    path,
    epoch,
):

    """
    Save the complete training state
    """

    check_point = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss
    }

    torch.save(
        check_point,
        path
    )

# load checkpoint function
def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device
):

    """
    Load the complete training state
    """
    
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_val_loss = checkpoint["best_val_loss"]

    return start_epoch, best_val_loss

# step 4 - train.py