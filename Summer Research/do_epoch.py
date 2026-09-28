import torch


def do_epoch(model, loader, loss_function, device, train=True, optimizer=None):
    running_loss = 0
    correct = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)

        # zero gradients in training
        if train:
            optimizer.zero_grad()

        # get outputs and loss
        outputs = model(data)
        labels = labels.long() # FOR CLASSIFICATION
        loss = loss_function(outputs, labels)

        # compute gradients and update weights in training
        # otherwise, compute loss and num correct
        if train:
            loss.backward()
            optimizer.step()
        else:
            running_loss += loss.item() * data.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).float().sum()

    if not train:
        return running_loss, correct