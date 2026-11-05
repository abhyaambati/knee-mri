"""Rule-based knee MSK structure extractor.

Chest labelers such as CheXbert do not cover cruciate ligaments, menisci, or
compartmental cartilage. This parser reads a report and assigns one status to
each knee structure. Mention matching covers English and the other languages
present in the RSNA knee MRI reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

LIGAMENT_STATUSES = ("Intact", "Partial Tear", "Full-Thickness Tear", "Not Mentioned")
MENISCUS_STATUSES = ("Intact", "Degenerative / Signal Change", "Tear", "Not Mentioned")
CARTILAGE_STATUSES = ("Normal", "Defect / Chondromalacia", "Not Mentioned")
BINARY_STATUSES = ("Present", "Absent", "Not Mentioned")

LIGAMENTS = ("ACL", "PCL", "MCL", "LCL")
MENISCI = ("Medial Meniscus", "Lateral Meniscus")
CARTILAGE = ("Patellar Cartilage", "Trochlear Cartilage", "Tibiofemoral Cartilage")
BINARY = ("Joint Effusion", "Bone Marrow Edema")
STRUCTURES: Tuple[str, ...] = LIGAMENTS + MENISCI + CARTILAGE + BINARY

NOT_MENTIONED = "Not Mentioned"

# Higher rank wins when one report sentence upgrades a structure.
_RANK: Dict[str, int] = {
    "Not Mentioned": 0,
    "Intact": 1,
    "Normal": 1,
    "Absent": 1,
    "Degenerative / Signal Change": 2,
    "Partial Tear": 3,
    "Present": 3,
    "Defect / Chondromalacia": 3,
    "Full-Thickness Tear": 4,
    "Tear": 4,
}

_PATHOLOGY = {
    "Partial Tear",
    "Full-Thickness Tear",
    "Tear",
    "Degenerative / Signal Change",
    "Defect / Chondromalacia",
    "Present",
}

_NEGATION = re.compile(
    r"\b(no|not|without|absent|negative|unremarkable|denies|sin|sans|kein|keine|"
    r"kein(?:e[mnrs]?)?|geen|yok|ohne|aucun|aucune|nema|bez|yoktur|saptanmam[ıi][sş]t[ıi]r|"
    r"няма|χωρίς|δεν)\b",
    re.IGNORECASE,
)

_STRUCTURE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "ACL": (
        r"\bacl\b",
        r"anterior cruciate",
        r"cruciate ant[eé]rieur",
        r"ligaments? croisés? ant[eé]rieur",
        r"cruzado anterior",
        r"vorderes kreuzband",
        r"voorste kruisband",
        r"[oö]n [cç]apraz",
        r"[cç]apraz\s+ve\s+yan\s+ba[gğ]",
        r"[cç]apraz\s+ba[gğ]",
        r"prednji kri[zž]ni",
        r"предни(?:я)?\s+кръстн",
        r"πρόσθι\w*\s*χιασ",
        r"κρυ?στα\w*",
    ),
    "PCL": (
        r"\bpcl\b",
        r"posterior cruciate",
        r"cruciate post[eé]rieur",
        r"ligaments? croisés? post[eé]rieur",
        r"cruzado posterior",
        r"hinteres kreuzband",
        r"achterste kruisband",
        r"arka [cç]apraz",
        r"[cç]apraz\s+ve\s+yan\s+ba[gğ]",
        r"[cç]apraz\s+ba[gğ]",
        r"stra[zž]nji kri[zž]ni",
        r"задни(?:я)?\s+кръстн",
    ),
    "MCL": (
        r"\bmcl\b",
        r"medial collateral",
        r"collateral m[eé]dial",
        r"colateral medial",
        r"inneres seitenband",
        r"mediale collaterale",
        r"mediaal collateraal",
        r"i[cç] yan ba[gğ]",
        r"[cç]apraz\s+ve\s+yan\s+ba[gğ]",
        r"yan ba[gğ]",
        r"medijaln\w* kolateral",
        r"медиа\w*\s+колатерал",
        r"έσω\s*πλάγι",
    ),
    "LCL": (
        r"\blcl\b",
        r"lateral collateral",
        r"collateral lat[eé]ral",
        r"colateral lateral",
        r"[aä]u[sß]eres seitenband",
        r"laterale collaterale",
        r"lateraal collateraal",
        r"d[ıi][sş] yan ba[gğ]",
        r"[cç]apraz\s+ve\s+yan\s+ba[gğ]",
        r"yan ba[gğ]",
        r"lateraln\w* kolateral",
        r"латера\w*\s+колатерал",
        r"έξω\s*πλάγι",
    ),
    "Medial Meniscus": (
        r"medial menisc\w*",
        r"menisco (?:interno|medial)",
        r"m[eé]nisque (?:interne|m[eé]dial)",
        r"medialer meniskus",
        r"mediale meniscus",
        r"i[cç] menisk\w*",
        r"medyal\s+menisk",
        r"medyal\s+ve\s+lateral\s+menisk",
        r"menisk[uü]s[uü]n? i[cç]",
        r"medijaln\w* menisk",
        r"медиа\w*\s+мениск",
        r"έσω\s*μηνίσκ",
        r"μηνίσκ\w*\s*έσω",
    ),
    "Lateral Meniscus": (
        r"lateral menisc\w*",
        r"menisco (?:externo|lateral)",
        r"m[eé]nisque (?:externe|lat[eé]ral)",
        r"lateraler meniskus",
        r"laterale meniscus",
        r"d[ıi][sş] menisk\w*",
        r"lateral\s+menisk",
        r"medyal\s+ve\s+lateral\s+menisk",
        r"lateraln\w* menisk",
        r"латера\w*\s+мениск",
        r"έξω\s*μηνίσκ",
        r"μηνίσκ\w*\s*έξω",
    ),
    "Patellar Cartilage": (
        r"patell(?:ar|ofemoral)\s+(?:cartilage|chondral|articular)",
        r"cart[ií]lago (?:patelar|rotuliano|de la r[oó]tula)",
        r"chondromalacia patell",
        r"condromalacia rotul",
        r"retropatellar",
        r"kniescheibenknorpel",
        r"knorpelschaden (?:der )?patella",
        r"patellofemor\w*",
        r"femoropatell\w*",
        r"hrskavic\w* (?:patele|patel)",
        r"хрущ\w*\s+на\s+пател",
        r"επιγονατίδ",
    ),
    "Trochlear Cartilage": (
        r"trochlea(?:r)?\s+(?:cartilage|chondral|groove)",
        r"cart[ií]lago troclear",
        r"trochlear cartilage",
        r"trochlea(?:r)?knorpel",
        r"τροχιλί",
    ),
    "Tibiofemoral Cartilage": (
        r"tibiofemoral\s+(?:cartilage|chondral|compartment)",
        r"femorotibial\s+(?:cartilage|chondral|compartment)",
        r"cart[ií]lago (?:tibiofemoral|femorotibial)",
        r"weight[- ]bearing cartilage",
        r"medial compartment cartilage",
        r"lateral compartment cartilage",
        r"medial compartment",
        r"lateral compartment",
        r"articul(?:ar|ation)\s+cartilage",
        r"femoral(?:e|en)?\s+(?:i\s+)?tibijaln\w* hrskavic",
        r"ставн\w*\s+хрущ",
        r"αρθρικός\s+χόνδρος",
        r"eklem k[ıi]k[ıi]rdak",
    ),
    "Joint Effusion": (
        r"joint effusion",
        r"knee effusion",
        r"\beffusion\b",
        r"\bderrame\b",
        r"derrame articular",
        r"[eé]panchement",
        r"gelenkerguss",
        r"gelenkserguss",
        r"\bef[uü]zyon\b",
        r"mayii art[ıi][sş]",
        r"s[ıi]v[ıi] art[ıi][sş]",
        r"diz eklemi i[cç]i s[ıi]v[ıi]",
        r"intra-?articular fluid",
        r"\bizljev\b",
        r"nema izljeva",
        r"\bhydrops\b",
        r"\bставен излив\b",
        r"няма\s+данни\s+за\s+ставен\s+излив",
        r"ενδαρθρικ\w*\s*συλλογ",
        r"ύδραρθρο",
    ),
    "Bone Marrow Edema": (
        r"bone marrow (?:edema|oedema|contusion)",
        r"bone bruise",
        r"marrow edema",
        r"edema [oó]seo",
        r"contusi[oó]n [oó]sea",
        r"edema de m[eé]dula",
        r"knochenmarks[oö]dem",
        r"\bbone bruise\b",
        r"botoedeem",
        r"kemik ili[gğ]i [oö]dem",
        r"subchondral edema",
        r"ko[sš]tani edem",
        r"edem zglobnih",
        r"контузион\w*\s+костномоз",
        r"костномозъчен едем",
        r"οστικ\w*\s*μυελ",
    ),
}

_COMPILED = {
    name: re.compile("|".join(patterns), re.IGNORECASE) for name, patterns in _STRUCTURE_PATTERNS.items()
}

_PARTIAL = re.compile(
    r"partial(?:-thickness)?(?: tear| rupture)?|high-grade sprain|grade\s*(?:1|2|i{1,2})\b|"
    r"parcial|partielle|gedeeltelijke|k[ıi]smi|distensi[oó]n|sprain|esguince|"
    r"parcijaln\w*\s+ruptur|manja\s+parcijaln",
    re.IGNORECASE,
)
_FULL = re.compile(
    r"full-thickness|complete tear|complete rupture|\brupture\b|\btorn\b|\btear\b|"
    r"rotura|ruptura|desgarro|fissure|riss|scheur|y[ıi]rt[ıi]k|zerrei[sß]ung|"
    r"\bruptur[ae]\b",
    re.IGNORECASE,
)
_DEGENERATIVE = re.compile(
    r"degenerativ\w*|mucoid|intrasubstance|signal change|se[nñ]al degenerativa|"
    r"d[eé]g[eé]n[eé]ratif|dejeneratif",
    re.IGNORECASE,
)
_INTACT = re.compile(
    r"\bintact\b|\bpreserved\b|\bnormal\b|unremarkable|sin rotura|intacto|conservado|"
    r"unauff[aä]llig|intakt|sans rupture|sa[gğ]lam|no tear|without tear|"
    r"bez znakova|uredn\w*|nema znakova|normal(?:dir)?|korunmu[sş]",
    re.IGNORECASE,
)
_DEFECT = re.compile(
    r"chondromalacia|chondropathy|chondral (?:defect|loss|fissure|thinning|ulcer)|"
    r"cartilage (?:defect|loss|thinning)|condromalacia|knorpelschaden|"
    r"k[ıi]k[ıi]rdak (?:hasar|defekt)|full-thickness cartilage",
    re.IGNORECASE,
)
_PRESENT = re.compile(
    r"\b(small|moderate|large|mild|trace|minimal|present|tiny|physiologic|marked)\b|"
    r"m[ií]nim[oa]|leve|pequen[oa]|vorhanden|gering|petit|hafif|"
    r"contusi[oó]n|edema|edem|[oö]dem|botoedeem",
    re.IGNORECASE,
)


def empty_status() -> Dict[str, str]:
    """Every structure starts as not mentioned."""
    return {structure: NOT_MENTIONED for structure in STRUCTURES}


def _clauses(sentence: str) -> List[str]:
    parts = re.split(r"\s+(?:and|y|und|et)\s+|;\s*", sentence)
    return [part.strip() for part in parts if part.strip()]


def _affirmed(pattern: re.Pattern, text: str) -> bool:
    """True when a cue appears and a negation word does not govern it."""
    match = pattern.search(text)
    if match is None:
        return False
    window = text[max(0, match.start() - 48) : match.start()]
    return _NEGATION.search(window) is None


def _group_status(structure: str, text: str) -> str:
    negated = _NEGATION.search(text) is not None
    if structure in LIGAMENTS:
        if _affirmed(_PARTIAL, text):
            return "Partial Tear"
        if _affirmed(_FULL, text):
            return "Full-Thickness Tear"
        if _INTACT.search(text) or negated:
            return "Intact"
        return NOT_MENTIONED
    if structure in MENISCI:
        if _affirmed(_FULL, text):
            return "Tear"
        if _affirmed(_DEGENERATIVE, text):
            return "Degenerative / Signal Change"
        if _INTACT.search(text) or negated:
            return "Intact"
        return NOT_MENTIONED
    if structure in CARTILAGE:
        if _affirmed(_DEFECT, text):
            return "Defect / Chondromalacia"
        if _INTACT.search(text) or negated:
            return "Normal"
        return NOT_MENTIONED
    if _affirmed(_PRESENT, text):
        return "Present"
    if negated or _INTACT.search(text):
        return "Absent"
    # Naming the structure without a negative cue is itself a positive finding
    # for effusion / bone-marrow edema (e.g. "Derrame articular.").
    return "Present"


def _mentioned(structure: str, text: str) -> bool:
    if _COMPILED[structure].search(text):
        return True
    lowered = text.lower()
    if structure in ("ACL", "PCL") and re.search(
        r"cruciate ligaments|ligamentos cruzados|kreuzb[aä]nder|kri[zž]n\w* ligament|"
        r"kruisbanden|[cç]apraz(?:\s+ve\s+yan)?\s+ba[gğ]|[cç]apraz\s+ve\s+yan|"
        r"кръстн\w*\s+връз|χιαστ",
        lowered,
    ):
        return True
    if structure in ("MCL", "LCL") and re.search(
        r"collateral ligaments|ligamentos colaterales|kolateraln\w* ligament|"
        r"[cç]apraz\s+ve\s+yan\s+ba[gğ]|yan ba[gğ]|πλάγι\w*\s*σύνδεσμ|колатерал",
        lowered,
    ):
        return True
    if structure in MENISCI and re.search(
        r"\bmenisci\b|ambos meniscos|both menisci|medijaln\w* i lateraln\w* menisk|"
        r"medyal\s+ve\s+lateral\s+menisk|dva(?:ta)? menisk|μηνίσκοι|менискус",
        lowered,
    ):
        return True
    return False


def _assign(status: Dict[str, str], text: str) -> None:
    hits = [structure for structure in STRUCTURES if _mentioned(structure, text)]
    if not hits:
        return
    for structure in hits:
        candidate = _group_status(structure, text)
        if _RANK[candidate] >= _RANK[status[structure]]:
            status[structure] = candidate


@dataclass
class MSKComparison:
    """Per-report agreement between a reference status vector and a prediction."""

    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    flips: List[str]
    laterality_errors: List[str]
    hallucinations: List[str]
    omissions: List[str]


class MSKExtractor:
    """Extract knee structure statuses and score them as a multi-label set."""

    def extract(self, report: str) -> Dict[str, str]:
        status = empty_status()
        text = "" if report is None else str(report)
        if not text.strip():
            return status
        sentences = re.split(r"[\n\r]+|(?<=[\.\!\?])\s+", text)
        shared = re.compile(
            r"menisci|cruciate ligaments|collateral ligaments|ligamentos cruzados|"
            r"ligamentos colaterales|ambos meniscos|both menisci",
            re.IGNORECASE,
        )
        for sentence in sentences:
            if not sentence.strip():
                continue
            if shared.search(sentence):
                _assign(status, sentence)
                continue
            clauses = _clauses(sentence)
            if len(clauses) == 1:
                _assign(status, sentence)
                continue
            for clause in clauses:
                _assign(status, clause)
        return status

    def compare(self, reference: str, prediction: str) -> MSKComparison:
        ref_status = self.extract(reference)
        pred_status = self.extract(prediction)
        return compare_statuses(ref_status, pred_status)

    def corpus_scores(
        self, references: Sequence[str], predictions: Sequence[str]
    ) -> Dict[str, float]:
        if len(references) != len(predictions):
            raise ValueError("references and predictions must have the same length")
        comparisons = [self.compare(reference, prediction) for reference, prediction in zip(references, predictions)]
        true_positive = sum(item.true_positive for item in comparisons)
        false_positive = sum(item.false_positive for item in comparisons)
        false_negative = sum(item.false_negative for item in comparisons)
        micro = _prf(true_positive, false_positive, false_negative)
        per_label = _macro_f1(references, predictions, self)
        return {
            "precision": micro[0],
            "recall": micro[1],
            "micro_f1": micro[2],
            "macro_f1": per_label,
            "n": float(len(references)),
        }


def compare_statuses(reference: Dict[str, str], prediction: Dict[str, str]) -> MSKComparison:
    """Score one pair. Structures omitted on both sides do not count as matches."""
    true_positive = false_positive = false_negative = 0
    flips: List[str] = []
    hallucinations: List[str] = []
    omissions: List[str] = []
    for structure in STRUCTURES:
        ref_state = reference.get(structure, NOT_MENTIONED)
        pred_state = prediction.get(structure, NOT_MENTIONED)
        if ref_state == NOT_MENTIONED and pred_state == NOT_MENTIONED:
            continue
        if ref_state == pred_state:
            true_positive += 1
        else:
            if pred_state != NOT_MENTIONED:
                false_positive += 1
            if ref_state != NOT_MENTIONED:
                false_negative += 1
            if ref_state != NOT_MENTIONED and pred_state != NOT_MENTIONED:
                flips.append(f"{structure}: {ref_state} -> {pred_state}")
        if pred_state in _PATHOLOGY and ref_state not in _PATHOLOGY:
            hallucinations.append(structure)
        if ref_state in _PATHOLOGY and pred_state not in _PATHOLOGY:
            omissions.append(structure)
    precision, recall, f1 = _prf(true_positive, false_positive, false_negative)
    return MSKComparison(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        flips=flips,
        laterality_errors=_laterality_errors(reference, prediction),
        hallucinations=hallucinations,
        omissions=omissions,
    )


def _laterality_errors(reference: Dict[str, str], prediction: Dict[str, str]) -> List[str]:
    errors: List[str] = []
    pairs = (("Medial Meniscus", "Lateral Meniscus"), ("MCL", "LCL"))
    for left, right in pairs:
        ref_left = reference.get(left, NOT_MENTIONED) in _PATHOLOGY
        ref_right = reference.get(right, NOT_MENTIONED) in _PATHOLOGY
        pred_left = prediction.get(left, NOT_MENTIONED) in _PATHOLOGY
        pred_right = prediction.get(right, NOT_MENTIONED) in _PATHOLOGY
        if ref_left and not ref_right and pred_right and not pred_left:
            errors.append(f"{left} pathology was attributed to {right}")
        if ref_right and not ref_left and pred_left and not pred_right:
            errors.append(f"{right} pathology was attributed to {left}")
    return errors


def _prf(true_positive: int, false_positive: int, false_negative: int) -> Tuple[float, float, float]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _macro_f1(references: Sequence[str], predictions: Sequence[str], extractor: MSKExtractor) -> float:
    """Unweighted mean of per-status F1, ignoring statuses that never occur."""
    observed = set()
    ref_rows = []
    pred_rows = []
    for reference, prediction in zip(references, predictions):
        ref_rows.append(extractor.extract(reference))
        pred_rows.append(extractor.extract(prediction))
        observed.update(status for status in ref_rows[-1].values() if status != NOT_MENTIONED)
        observed.update(status for status in pred_rows[-1].values() if status != NOT_MENTIONED)
    scores: List[float] = []
    for label in sorted(observed):
        true_positive = false_positive = false_negative = 0
        for reference, prediction in zip(ref_rows, pred_rows):
            for structure in STRUCTURES:
                ref_hit = reference[structure] == label
                pred_hit = prediction[structure] == label
                if ref_hit and pred_hit:
                    true_positive += 1
                elif pred_hit:
                    false_positive += 1
                elif ref_hit:
                    false_negative += 1
        scores.append(_prf(true_positive, false_positive, false_negative)[2])
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
