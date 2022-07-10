from math import log2
from torch import Tensor, sort
import torch
import math

def num_swapped_pairs(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> int:
    """
    функция для расчёта количества неправильно упорядоченных пар
    (корректное упорядочивание — от наибольшего значения в ys_true к наименьшему)
    или переставленных пар. Не забудьте, что одну и ту же пару не нужно учитывать дважды.  
    """
    ys_pred_sorted, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    
    num_objects = ys_true_sorted.shape[0]
    swapped_cnt = 0
    for cur_obj in range(num_objects - 1):
        for next_obj in range(cur_obj + 1, num_objects):
            if ys_true_sorted[cur_obj] < ys_true_sorted[next_obj]:
                if ys_pred_sorted[cur_obj] > ys_pred_sorted[next_obj]:
                    swapped_cnt += 1
            elif ys_true_sorted[cur_obj] > ys_true_sorted[next_obj]:
                if ys_pred_sorted[cur_obj] < ys_pred_sorted[next_obj]:
                    swapped_cnt += 1
    return swapped_cnt


def compute_gain(y_value: float, gain_scheme: str) -> float:
    """
    compute_gain — вспомогательная функция для расчёта DCG и NDCG,
    рассчитывающая показатель Gain.
    Принимает на вход дополнительный аргумент — указание схемы начисления Gain (gain_scheme).
    """
    if gain_scheme == "exp2":
        gain = 2 ** y_value - 1
    elif gain_scheme == "const":
        gain = y_value
    else:
        raise ValueError("Invalid gains option.")
    return float(gain)


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    ret = 0
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        gain = compute_gain(cur_y, gain_scheme)
        ret += gain / math.log2(idx + 1)
    return ret


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    best = dcg(ys_true, ys_true, gain_scheme)
    actual = dcg(ys_true, ys_pred, gain_scheme)
    ndcg = actual / best
    return ndcg

def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    """
    функция расчёта точности в топ-k позиций для бинарной разметки
    (в ys_true содержатся только нули и единицы).
    """
    if ys_true.sum() == 0:
        return -1
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    # кол-во единичек (на входе идет 0 или 1)
    hits = ys_true_sorted[:k].sum()
    prec = hits / min(ys_true.sum(), k)
    return float(prec)


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    """
    функция для расчёта MRR (без усреднения,
    т.е. для одного запроса и множества документов).
    В ys_true могут содержаться только нули и максимум одна единица.
    """
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        if cur_y == 1:
            return 1 / idx
    return 0


def p_found(ys_true: torch.Tensor, ys_pred: torch.Tensor, p_break: float = 0.15 ) -> float:
    """
    функция расчёта P-found от Яндекса, принимающая на вход
    дополнительный параметр p_break — вероятность прекращения
    просмотра списка документов в выдаче.
    """
    p_look = 1
    p_found = 0
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]

    for cur_y in ys_true_sorted:
        p_found += p_look * float(cur_y)
        p_look = p_look * (1 - float(cur_y)) * (1 - p_break)
    
    return p_found


def average_precision(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    """
    функция расчёта AP для бинарной разметки (в ys_true содержатся только нули и единицы).
    Если среди лейблов нет ни одного релевантного документа (единицы),
    то необходимо вернуть -1.
    """
    if ys_true.sum() == 0:
        return -1
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    rolling_sum = 0
    num_correct_ans = 0
    
    for idx, cur_y in enumerate(ys_true_sorted, start=1):
        if cur_y == 1:
            num_correct_ans += 1
            rolling_sum += num_correct_ans / idx
    if num_correct_ans == 0:
        return 0
    else:
        return rolling_sum / num_correct_ans
