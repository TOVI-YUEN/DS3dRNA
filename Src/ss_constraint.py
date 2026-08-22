# Src/ss_constraint.py
# -*- coding: utf-8 -*-

import os
import string
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch


# ------------------------------------------------------------
# Bracket definitions
# ------------------------------------------------------------
_PUNCT_OPEN_TO_CLOSE = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
}

_LETTER_OPEN_TO_CLOSE = {
    up: low for up, low in zip(string.ascii_uppercase, string.ascii_lowercase)
}

OPEN_TO_CLOSE = dict(_PUNCT_OPEN_TO_CLOSE)
OPEN_TO_CLOSE.update(_LETTER_OPEN_TO_CLOSE)

CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}

SS_PUNCT_CHARS = set(".&()[]{}<>")
SS_ALL_CHARS = set(".&") | set(OPEN_TO_CLOSE.keys()) | set(CLOSE_TO_OPEN.keys())

SEQ_CHARS = set("ACGUTNIXacgutnix")


# ------------------------------------------------------------
# Homopolymer / poly-base penalty global parameters
# ------------------------------------------------------------
# Base code convention used by DS3dRNA:
#   A=0, U=1, C=2, G=3
#
# A run is penalized when its maximal within-chain length is >= threshold.
# Set a threshold <= 0 or None to disable that base-specific poly penalty.
POLY_A_MIN_LEN = 6
POLY_U_MIN_LEN = 6
POLY_C_MIN_LEN = 6
POLY_G_MIN_LEN = 6

POLY_BASE_MIN_LEN = {
    0: POLY_A_MIN_LEN,  # A
    1: POLY_U_MIN_LEN,  # U
    2: POLY_C_MIN_LEN,  # C
    3: POLY_G_MIN_LEN,  # G
}

POLY_BASE_NAME = {
    0: "A",
    1: "U",
    2: "C",
    3: "G",
}


def _poly_rule_enabled(min_len: Optional[int]) -> bool:
    return min_len is not None and int(min_len) > 0


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _strip_all_whitespace(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def effective_ss_length(ss: str) -> int:
    s = _strip_all_whitespace(ss)
    return sum(1 for ch in s if ch != "&")


def compact_ss_string(ss: str) -> str:
    s = _strip_all_whitespace(ss)
    return s.replace("&", "")


def _is_probable_sequence_line(line: str) -> bool:
    s = _strip_all_whitespace(line)
    if not s:
        return False

    s2 = s.replace("&", "")
    if not s2:
        return False

    return all(ch in SEQ_CHARS for ch in s2)


def _is_probable_ss_line(line: str) -> bool:
    s = _strip_all_whitespace(line)
    if not s:
        return False

    if not all(ch in SS_ALL_CHARS for ch in s):
        return False

    if all(ch in SS_PUNCT_CHARS for ch in s):
        return True

    if _is_probable_sequence_line(s):
        return False

    has_punct_bracket = any(ch in "()[]{}<>" for ch in s)
    has_lower = any(ch.islower() for ch in s)

    return has_punct_bracket or has_lower


def _build_chain_segments(raw_ss: str) -> List[Tuple[int, int]]:
    """
    Build compact-index chain segments [start, end) from raw ss string.
    '&' splits chains and does NOT consume residue index.
    """
    s = _strip_all_whitespace(raw_ss)
    segs: List[Tuple[int, int]] = []

    start = 0
    cur = 0
    in_chain = False

    for ch in s:
        if ch == "&":
            if in_chain:
                segs.append((start, cur))
                in_chain = False
            continue

        if not in_chain:
            start = cur
            in_chain = True

        cur += 1

    if in_chain:
        segs.append((start, cur))

    return segs


def _build_stem_terminal_pairs(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Group consecutive stacked base pairs into stems, then return terminal pairs
    of each stem.

    Stacking rule:
      (i, j) -> (i+1, j-1)

    For each stem:
      - length == 1 : its only pair is counted once
      - length >= 2 : count first and last pair as terminal pairs

    NOTE:
      This helper is preserved for compatibility with old code paths.
      The current constraint no longer applies any special terminal CG clamp.
    """
    if not pairs:
        return []

    pair_set = set(pairs)
    terminals: List[Tuple[int, int]] = []

    for i, j in pairs:
        if (i - 1, j + 1) in pair_set:
            continue

        stem = []
        a, b = i, j
        while (a, b) in pair_set:
            stem.append((a, b))
            a += 1
            b -= 1

        if len(stem) == 1:
            terminals.append(stem[0])
        else:
            terminals.append(stem[0])
            terminals.append(stem[-1])

    return terminals


def _build_valid_run_starts(
    compact_len: int,
    chain_segments: List[Tuple[int, int]],
    min_len: int,
    device: torch.device,
):
    """
    Precompute valid homopolymer-run starts that do NOT cross '&'.

    Returns:
      valid_starts:
        shape (compact_len - min_len + 1,)
      prev_same_chain:
        shape (compact_len - min_len + 1,)
        whether previous residue exists in the same chain for maximal-run counting.
    """
    if compact_len < min_len:
        return None, None

    L = compact_len - min_len + 1
    valid_starts = torch.zeros(L, dtype=torch.bool, device=device)
    prev_same_chain = torch.zeros(L, dtype=torch.bool, device=device)

    for start, end in chain_segments:
        seg_len = end - start

        if seg_len >= min_len:
            valid_starts[start : end - min_len + 1] = True

        if seg_len >= min_len + 1:
            prev_same_chain[start + 1 : end - min_len + 1] = True

    return valid_starts, prev_same_chain


def sample_penalty_free_sequence(
    ss_constraint: "SSConstraint",
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    pair_weights: Optional[torch.Tensor] = None,
    batch_size: int = 320,
    max_rounds: int = 8,
) -> torch.Tensor:
    """
    Batched rejection sampler for one zero-penalty sequence.

    Base code convention:
      A=0, U=1, C=2, G=3

    Pair legality:
      AU, UA, CG, GC, UG, GU

    Extra enforced heuristics:
      - no terminal CG clamp; terminal pairs follow the same legality rule
        as all other required pairs
      - no within-chain homopolymer runs above the global thresholds:
        POLY_A_MIN_LEN, POLY_U_MIN_LEN, POLY_C_MIN_LEN, POLY_G_MIN_LEN
    """
    if ss_constraint is None:
        raise ValueError("sample_penalty_free_sequence: ss_constraint is None")

    out_device = ss_constraint.device if device is None else torch.device(device)
    N = int(ss_constraint.compact_len)

    if N <= 0:
        raise ValueError(f"sample_penalty_free_sequence: invalid compact_len={N}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if max_rounds <= 0:
        raise ValueError(f"max_rounds must be > 0, got {max_rounds}")

    allowed_pairs = torch.tensor(
        [
            [0, 1],  # A-U
            [1, 0],  # U-A
            [2, 3],  # C-G
            [3, 2],  # G-C
            [1, 3],  # U-G
            [3, 1],  # G-U
        ],
        dtype=torch.long,
        device=out_device,
    )

    if pair_weights is not None:
        if not isinstance(pair_weights, torch.Tensor):
            pair_weights = torch.as_tensor(
                pair_weights,
                dtype=torch.float32,
                device=out_device,
            )
        else:
            pair_weights = pair_weights.to(device=out_device, dtype=torch.float32)

        if pair_weights.shape != (6,):
            raise ValueError(
                f"pair_weights must have shape (6,), got shape={tuple(pair_weights.shape)}"
            )
        if torch.any(pair_weights < 0):
            raise ValueError("pair_weights must be non-negative")

        s = float(pair_weights.sum().item())
        if s <= 0.0:
            raise ValueError("pair_weights sum must be > 0")

        pair_probs = pair_weights / pair_weights.sum()
    else:
        pair_probs = None

    if ss_constraint.pairs_i is not None:
        pair_i = ss_constraint.pairs_i.to(out_device)
        pair_j = ss_constraint.pairs_j.to(out_device)
    else:
        pair_i = None
        pair_j = None

    for _ in range(max_rounds):
        # Free positions start from U/C only to strongly suppress accidental
        # long G-runs and poly-base runs in the rejection sampler.
        seq_batch = torch.randint(
            low=1,
            high=3,   # {U, C}
            size=(batch_size, N),
            dtype=torch.long,
            device=out_device,
            generator=generator,
        )

        # All required pairs, including stem-terminal pairs, are sampled
        # from the same allowed-pair distribution:
        # AU, UA, CG, GC, UG, GU.
        #
        # No special terminal CG clamp is applied.
        if pair_i is not None:
            n_pairs = int(pair_i.numel())

            if pair_probs is None:
                k = torch.randint(
                    low=0,
                    high=6,
                    size=(batch_size, n_pairs),
                    device=out_device,
                    generator=generator,
                )
            else:
                k = torch.multinomial(
                    pair_probs,
                    num_samples=batch_size * n_pairs,
                    replacement=True,
                    generator=generator,
                ).view(batch_size, n_pairs)

            sampled = allowed_pairs[k]  # (B, K, 2)
            seq_batch[:, pair_i] = sampled[:, :, 0]
            seq_batch[:, pair_j] = sampled[:, :, 1]

        eval_batch = seq_batch if out_device == ss_constraint.device else seq_batch.to(ss_constraint.device)
        pen = ss_constraint.score_batch(eval_batch)
        ok = (pen == 0)

        if torch.any(ok):
            idx = int(torch.nonzero(ok, as_tuple=True)[0][0].detach().cpu().item())
            return seq_batch[idx]

    raise RuntimeError(
        "sample_penalty_free_sequence: failed to sample a zero-penalty sequence "
        f"after {batch_size * max_rounds} parallel attempts"
    )


# ------------------------------------------------------------
# Public validators / loaders
# ------------------------------------------------------------
def looks_like_dotbracket(ss: str) -> bool:
    if ss is None:
        return False
    return _is_probable_ss_line(ss)


def load_ss_string(ss_arg: Optional[str]) -> Optional[str]:
    if ss_arg is None:
        return None

    ss_arg = str(ss_arg).strip()
    if not ss_arg:
        return None

    if os.path.isfile(ss_arg):
        with open(ss_arg, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]

        for line in reversed(lines):
            line_s = _strip_all_whitespace(line)
            if not line_s:
                continue
            if line_s.startswith(">"):
                continue
            if _is_probable_ss_line(line_s):
                return line_s

        raise ValueError(f"No valid secondary-structure line found in file: {ss_arg}")

    ss_s = _strip_all_whitespace(ss_arg)
    if not ss_s:
        raise ValueError("Empty --ss input")

    if not _is_probable_ss_line(ss_s):
        raise ValueError(
            f"Invalid --ss input. It is neither a valid structure string nor a readable dbn file: {ss_arg}"
        )

    return ss_s


# ------------------------------------------------------------
# Parsing
# ------------------------------------------------------------
def parse_ss_to_pairs(ss: str) -> List[Tuple[int, int]]:
    """
    Parse raw ss string into 0-based residue-index pairs.

    Supported:
      - standard dot-bracket: () [] {} <>
      - extended letter-based pseudoknot notation: A..a, B..b, ...
      - '&' as chain separator

    IMPORTANT:
      '&' does NOT count as a residue index.
    """
    if not isinstance(ss, str) or len(ss) == 0:
        raise ValueError("Empty secondary structure string")

    s = _strip_all_whitespace(ss)

    stacks = {op: [] for op in OPEN_TO_CLOSE.keys()}
    pairs: List[Tuple[int, int]] = []

    res_idx = 0

    for raw_idx, ch in enumerate(s):
        if ch == "&":
            continue

        if ch in OPEN_TO_CLOSE:
            stacks[ch].append(res_idx)
            res_idx += 1

        elif ch in CLOSE_TO_OPEN:
            op = CLOSE_TO_OPEN[ch]
            if not stacks[op]:
                raise ValueError(
                    f"Unmatched closing bracket '{ch}' at raw char position {raw_idx}"
                )
            i = stacks[op].pop()
            pairs.append((i, res_idx))
            res_idx += 1

        elif ch == ".":
            res_idx += 1

        else:
            raise ValueError(
                f"Unsupported SS character '{ch}' at raw char position {raw_idx}"
            )

    for op, st in stacks.items():
        if st:
            raise ValueError(
                f"Unmatched opening bracket '{op}' at residue positions {st}"
            )

    pairs.sort()

    used = set()
    for i, j in pairs:
        if i == j:
            raise ValueError(f"Invalid self-pair detected: ({i}, {j})")
        if i in used or j in used:
            raise ValueError(
                "Invalid SS: a residue appears in more than one base pair. "
                f"Offending pair=({i}, {j})"
            )
        used.add(i)
        used.add(j)

    return pairs


# ------------------------------------------------------------
# Constraint object
# ------------------------------------------------------------
@dataclass
class SSConstraint:
    raw_ss_string: str
    compact_len: int
    pairs: List[Tuple[int, int]]
    penalty: float
    device: torch.device

    def __post_init__(self):
        # A=0, U=1, C=2, G=3
        # pair_code = left * 4 + right
        allowed = torch.zeros(16, dtype=torch.bool, device=self.device)

        allowed[0 * 4 + 1] = True  # A-U
        allowed[1 * 4 + 0] = True  # U-A
        allowed[2 * 4 + 3] = True  # C-G
        allowed[3 * 4 + 2] = True  # G-C
        allowed[1 * 4 + 3] = True  # U-G
        allowed[3 * 4 + 1] = True  # G-U
        self.allowed = allowed

        self.chain_segments = _build_chain_segments(self.raw_ss_string)

        # Precompute valid starts that do NOT cross '&' for each base-specific
        # homopolymer threshold. The thresholds are controlled by global params:
        #   POLY_A_MIN_LEN, POLY_U_MIN_LEN, POLY_C_MIN_LEN, POLY_G_MIN_LEN
        self.poly_valid_starts = {}
        self.poly_prev_same_chain = {}
        for base_code, min_len in POLY_BASE_MIN_LEN.items():
            if not _poly_rule_enabled(min_len):
                self.poly_valid_starts[base_code] = None
                self.poly_prev_same_chain[base_code] = None
                continue

            valid_starts, prev_same_chain = _build_valid_run_starts(
                compact_len=self.compact_len,
                chain_segments=self.chain_segments,
                min_len=int(min_len),
                device=self.device,
            )
            self.poly_valid_starts[base_code] = valid_starts
            self.poly_prev_same_chain[base_code] = prev_same_chain

        if len(self.pairs) == 0:
            self.pairs_i = None
            self.pairs_j = None

            # Preserved for compatibility with old external code paths.
            # They are no longer used for terminal CG clamp scoring.
            self.terminal_i = None
            self.terminal_j = None
            self.terminal_i_out = None
            self.terminal_j_out = None

            self.nonterminal_i = None
            self.nonterminal_j = None
            return

        self.pairs_i = torch.tensor(
            [i for i, _ in self.pairs],
            dtype=torch.long,
            device=self.device,
        )
        self.pairs_j = torch.tensor(
            [j for _, j in self.pairs],
            dtype=torch.long,
            device=self.device,
        )

        # Terminal / nonterminal bookkeeping is preserved for compatibility.
        # The current constraint no longer applies terminal-specific CG clamp.
        terminal_pairs = _build_stem_terminal_pairs(self.pairs)
        terminal_set = set(terminal_pairs)
        nonterminal_pairs = [p for p in self.pairs if p not in terminal_set]

        if len(terminal_pairs) == 0:
            self.terminal_i = None
            self.terminal_j = None
            self.terminal_i_out = None
            self.terminal_j_out = None
        else:
            t_i = torch.tensor(
                [i for i, _ in terminal_pairs],
                dtype=torch.long,
                device=self.device,
            )
            t_j = torch.tensor(
                [j for _, j in terminal_pairs],
                dtype=torch.long,
                device=self.device,
            )

            self.terminal_i = t_i
            self.terminal_j = t_j

            self.terminal_i_out = t_i
            self.terminal_j_out = t_j

        if len(nonterminal_pairs) == 0:
            self.nonterminal_i = None
            self.nonterminal_j = None
        else:
            self.nonterminal_i = torch.tensor(
                [i for i, _ in nonterminal_pairs],
                dtype=torch.long,
                device=self.device,
            )
            self.nonterminal_j = torch.tensor(
                [j for _, j in nonterminal_pairs],
                dtype=torch.long,
                device=self.device,
            )

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def _count_maximal_runs_ge(
        self,
        is_base: torch.Tensor,
        min_len: int,
        valid_starts: Optional[torch.Tensor],
        prev_same_chain: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Count maximal within-chain homopolymer runs with length >= min_len.

        Each maximal run contributes exactly 1 count.
        """
        B, N = is_base.shape

        if (
            valid_starts is None
            or valid_starts.numel() == 0
            or N < min_len
        ):
            return torch.zeros((B,), dtype=torch.float32, device=is_base.device)

        L = N - min_len + 1

        if min_len == 4:
            run_mask = (
                is_base[:, :-3]
                & is_base[:, 1:-2]
                & is_base[:, 2:-1]
                & is_base[:, 3:]
            )
        elif min_len == 5:
            run_mask = (
                is_base[:, :-4]
                & is_base[:, 1:-3]
                & is_base[:, 2:-2]
                & is_base[:, 3:-1]
                & is_base[:, 4:]
            )
        else:
            run_mask = is_base[:, :L].clone()
            for off in range(1, min_len):
                run_mask &= is_base[:, off : off + L]

        run_mask = run_mask & valid_starts.unsqueeze(0)

        if run_mask.shape[1] == 0:
            return torch.zeros((B,), dtype=torch.float32, device=is_base.device)

        prev_base = torch.zeros_like(run_mask)
        if run_mask.shape[1] > 1:
            prev_base[:, 1:] = is_base[:, : N - min_len]

        if prev_same_chain is not None:
            prev_base = prev_base & prev_same_chain.unsqueeze(0)

        run_starts = run_mask & (~prev_base)
        return run_starts.sum(dim=1).to(torch.float32)

    def _score_terminal_clamp(self, seq_batch: torch.Tensor) -> torch.Tensor:
        """
        Deprecated compatibility stub.

        Terminal CG clamp has been removed.
        Stem-terminal pairs now use the same legality rule as all other
        required pairs: AU, UA, CG, GC, UG, GU.

        This function intentionally returns zero.
        """
        B = seq_batch.shape[0]
        return torch.zeros((B,), dtype=torch.float32, device=seq_batch.device)

    def _score_homopolymer_runs(self, seq_batch: torch.Tensor) -> torch.Tensor:
        """
        Penalize within-chain homopolymer runs according to global thresholds:

          - A-run >= POLY_A_MIN_LEN
          - U-run >= POLY_U_MIN_LEN
          - C-run >= POLY_C_MIN_LEN
          - G-run >= POLY_G_MIN_LEN

        Each maximal run contributes exactly + self.penalty once.
        """
        B, N = seq_batch.shape

        enabled_rules = [
            (base_code, int(min_len))
            for base_code, min_len in POLY_BASE_MIN_LEN.items()
            if _poly_rule_enabled(min_len)
        ]

        if not enabled_rules:
            return torch.zeros((B,), dtype=torch.float32, device=seq_batch.device)

        min_required_len = min(min_len for _, min_len in enabled_rules)
        if N < min_required_len:
            return torch.zeros((B,), dtype=torch.float32, device=seq_batch.device)

        total = torch.zeros((B,), dtype=torch.float32, device=seq_batch.device)

        for base_code, min_len in enabled_rules:
            is_base = (seq_batch == int(base_code))
            valid_starts = self.poly_valid_starts.get(base_code)
            prev_same_chain = self.poly_prev_same_chain.get(base_code)

            total = total + self._count_maximal_runs_ge(
                is_base=is_base,
                min_len=int(min_len),
                valid_starts=valid_starts,
                prev_same_chain=prev_same_chain,
            )

        return total * float(self.penalty)

    def score_batch(self, seq_batch: torch.Tensor) -> torch.Tensor:
        """
        seq_batch:
          - (N,)
          - (B, N)

        returns:
          - (B,) tensor of penalties

        Penalty definition:
          1) each violated required pair adds `self.penalty`
          2) no terminal CG clamp is applied
          3) each maximal within-chain homopolymer run above the global
             base-specific threshold adds `self.penalty`:
             POLY_A_MIN_LEN, POLY_U_MIN_LEN, POLY_C_MIN_LEN, POLY_G_MIN_LEN
        """
        if not isinstance(seq_batch, torch.Tensor):
            seq_batch = torch.as_tensor(seq_batch, dtype=torch.long, device=self.device)
        else:
            seq_batch = seq_batch.to(self.device)

        if seq_batch.dim() == 1:
            seq_batch = seq_batch.unsqueeze(0)

        if seq_batch.dim() != 2:
            raise ValueError(
                f"seq_batch must be (N,) or (B,N), got shape={tuple(seq_batch.shape)}"
            )

        B, N = seq_batch.shape
        if int(N) != int(self.compact_len):
            raise ValueError(
                f"SS length mismatch in score_batch: seq_len={N} vs ss_compact_len={self.compact_len}"
            )

        pen = torch.zeros((B,), dtype=torch.float32, device=seq_batch.device)

        # 1) Required pair legality
        if self.pairs_i is not None:
            left = seq_batch[:, self.pairs_i]    # (B, K)
            right = seq_batch[:, self.pairs_j]   # (B, K)

            pair_code = left * 4 + right
            ok = self.allowed[pair_code]         # (B, K)
            bad = ~ok

            pen = pen + bad.sum(dim=1).to(torch.float32) * float(self.penalty)

        # 2) Homopolymer-run penalties
        pen = pen + self._score_homopolymer_runs(seq_batch)

        return pen


# ------------------------------------------------------------
# Builder
# ------------------------------------------------------------
def build_ss_constraint(
    ss_arg: Optional[str],
    native_len: int,
    device: torch.device,
    penalty: float = 10000.0,
) -> Optional[SSConstraint]:
    """
    Build a robust SSConstraint object.

    Length check is always done on the effective residue length:
      len(ss without '&' and whitespace) == native_len
    """
    if ss_arg is None:
        return None

    raw_ss = load_ss_string(ss_arg)
    if raw_ss is None:
        return None

    compact_len = effective_ss_length(raw_ss)
    if int(compact_len) != int(native_len):
        raise ValueError(
            f"SS length mismatch: effective_len(ss)={compact_len} vs native_len={int(native_len)}"
        )

    pairs = parse_ss_to_pairs(raw_ss)

    return SSConstraint(
        raw_ss_string=raw_ss,
        compact_len=int(compact_len),
        pairs=pairs,
        penalty=float(penalty),
        device=device,
    )
