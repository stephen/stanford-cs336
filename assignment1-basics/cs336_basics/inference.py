import argparse
import torch as t

from cs336_basics.softmax import softmax
from cs336_basics.tokenizer_cls import Tokenizer
from cs336_basics.trainer import ModelArgs
from cs336_basics.transformer import TransformerLM

default_device = t.device('mps') if t.backends.mps.is_available() else t.device('cuda') if t.cuda.is_available() else t.device('cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="./data/model.pth", help='Which file to use as the model weights')
    parser.add_argument('--tokenizer-state', type=str, default=None, help='Which files to use as the tokenizer serialized state')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens to generate for a response')
    parser.add_argument('--temperature', type=float, default=0, help='Softmax temperature when sampling outputs, or argmax if 0. Try 0.95 to start.')
    parser.add_argument('--top-p', type=float, default=1, help='P value for top-p (nucleus) sampling. Try 0.8 to start.')
    cli_args = parser.parse_args()

    assert cli_args.model, "--model must be specified"
    assert cli_args.tokenizer_state, "--tokenizer-state must be specified"

    model_args = ModelArgs()
    model = TransformerLM(
        context_len=model_args.context_len,
        d_ff=model_args.d_ff,
        d_model=model_args.d_model,
        n_heads=model_args.n_heads,
        n_layers=model_args.n_layers,
        rope_theta=model_args.rope_theta,
        vocab_size=model_args.vocab_size,
        device=default_device,
    )
    with open(cli_args.model, "rb") as f:
        model.load_state_dict(t.load(f))

    max_tokens = cli_args.max_tokens
    temperature = cli_args.temperature
    top_p = cli_args.top_p

    tokenizer = Tokenizer.from_file(cli_args.tokenizer_state)

    eot_encode = tokenizer.encode("<|endoftext|>")
    assert len(eot_encode) == 1, "expected only one token for <|endoftext|>"
    eot = eot_encode[0]

    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue

            generated = 0
            encoded = t.tensor(tokenizer.encode(line)).unsqueeze(0)
            while True:
                logits = model(encoded)
                if temperature == 0:
                    q = softmax(logits[-1][-1], dim=-1)
                    token_id = t.argmax(q, dim=-1, keepdim=True).item()
                else:
                    last_logits = logits[-1][-1] / temperature
                    q = softmax(last_logits, dim=-1)
                    token_id = t.multinomial(q, 1).long().item()

                if top_p != 1:
                    # {v: i for i, v in sorted(enumerate(q.tolist()), key=lambda x: x[1])}
                    sorted_probs, sorted_indices = t.sort(q, descending=True)
                    cumsum = t.cumsum(sorted_probs, dim=-1)
                    cutoff = t.searchsorted(cumsum, top_p, right=True) + 1

                    sorted_probs[cutoff] = 0
                    q = t.zeros_like(q).scatter_(0, sorted_indices, sorted_probs)
                    q = q / q.sum()
                    token_id = t.multinomial(q, 1).long().item()

                if token_id == eot:
                    break

                token = tokenizer.decode([int(token_id)])
                print(token, end="", flush=True)

                encoded = t.concat([encoded, t.tensor([token_id]).unsqueeze(0)], dim=-1)

                if encoded.shape[-1] > model_args.context_len:
                    encoded = encoded[..., -model_args.context_len:]

                generated += 1

                if generated >= max_tokens:
                    break

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        print("")

if __name__ == "__main__":
    main()
