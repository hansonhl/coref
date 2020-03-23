import torch
import logging
import tqdm
from anagen.dataset import collate

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

DEFAULT_TRAIN_BATCH_SIZE=1
logger = logging.getLogger(__name__)

def batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch

# based on transformers/run_lm_finetuning
def train(args, model, train_dataset, eval_dataset):
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
    num_batches = len(train_dataset)
    # start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_batches)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.train_batch_size)

    # TODO: add checkpoint loading functionality

    global_step = 0
    for epoch in range(args.num_train_epochs):
        print("*** Epoch %d ***" % epoch)
        for step, batch in enumerate(train_dataloader):
            batch = batch_to_device(batch, device)
            model.zero_grad()
            model.train()

            res_dict = model(batch, calculate_ppl=(global_step + 1) % args.train_log_steps == 0)
            loss = res_dict["loss"]

            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.train_log_steps == 0:
                print("  step %d/%d, global_step %d, batch loss = %.4f" \
                    % (step, num_batches, global_step, loss))

            if global_step % args.eval_and_save_steps == 0:
                eval_results = evaluate(args, model, eval_dataset)
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



def evaluate(args, model, eval_dataset):
    device = torch.device("cuda" if args.gpu else "cpu")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=DEFAULT_TRAIN_BATCH_SIZE,
                                 collate_fn=collate)

    print("***** Running evaluation *****")
    print("  Num examples = %d" % len(eval_dataset))
    print("  Batch size = %d" % args.eval_batch_size)

    eval_loss = 0.0
    num_toks = 0
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = batch_to_device(batch, device)
            res_dict = model(batch, calculate_ppl=True)
            eval_loss += res_dict["loss"].item() * num_toks
            num_toks += res_dict["num_toks"].item()

    eval_loss = eval_loss / num_toks
    perplexity = torch.exp(torch.tensor(eval_loss))

    print("***** Eval results *****")
    print("  eval_loss = %.6f" % eval_loss)
    print("  perplexity = %.6f" % perplexity)
