import torch
import logging
import tqdm
from anagen.dataset import collate

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

DEFAULT_TRAIN_BATCH_SIZE=1
logger = logging.getLogger(__name__)

# based on transformers/run_lm_finetuning
def train(args, model, train_dataset, eval_dataset,):
    device = torch.device("cuda" if args.gpu else "cpu")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=DEFAULT_TRAIN_BATCH_SIZE,
                                  collate_fn=collate)

    # not sure what the following does, copied from run_lm_finetuning code
    """
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps \
            // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // \
            args.gradient_accumulation_steps * args.num_train_epochs
    """
    model.freeze_gpt2()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.train_batch_size)

    # TODO: add train from checkpoint

    train_iterator = tqdm.trange(int(args.num_train_epochs), desc="Epoch")
    global_step = 0
    for epoch in train_iterator:
        for step, batch in enumerate(train_dataloader):
            batch_to_device(batch, device)
            model.zero_grad()
            model.train()

            scores, loss = model(batch)

            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.eval_and_save_steps == 0:
                print("  step %d, batch train loss = %.3f" % (global_step, loss))
                eval_results = evaluate(args, eval_dataset, model)
                # TODO: add tensorboard writer functionality

                if args.model_save_path:
                    print("  saving model to %s" % args.model_save_path)
                    model_checkpoint = {
                        "args": args,
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(), # just save everything for now
                        "optimizer_state_dict": optimizer.state_dict()
                    }
                    torch.save(model_checkpoint, args.model_save_path)


def batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch

def evaluate(args, eval_dataset, model):
    pass
