from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge


class Evaluator():
    """
    Evaluator is used to calculate NLP metrics over machine-generated captions
    y-hat of an image given the ground-truth reference captions y for that 
    image. 

    Methods
    -------
    update_scores(references, hypotheses)
        Calculates scores and updates running sum in dictionary for each method

    reset_scores
        resets scores to 0

    get_scores
        returns scores

    set_ckpt_scores(ckpt)
        sets ckpt.scores to Evaluator.scores
    """

    def __init__(self):
        self.iterations = 0
        self.scores = {
            'BLEU_1': 0,
            'BLEU_2': 0,
            'BLEU_3': 0,
            'BLEU_4': 0,
            'METEOR': 0,
            'ROUGE': 0,
            'CIDEr': 0,
        }
        self.B = Bleu(4)  # bleu1 to bleu4
        try:
            self.M = Meteor()
        except Exception:
            pass
        self.R = Rouge()  # Rouge-L
        self.C = Cider()

    def update_scores(self, references, hypotheses):
        """
        Calculates scores and updates running sum in dictionary for each method

        Parameters
        ----------
        references : list of lists of strings
            references where each inner-list is the ground truth y for each 
            hypothesis y-hat
        hypotheses : list of string 
            hypotheses y-hat
        """
        refs = {idx: sentences for (idx, sentences) in enumerate(references)}
        hyps = {idx: [sentence] for (idx, sentence) in enumerate(hypotheses)}
        [b1, b2, b3, b4], _ = self.B.compute_score(refs, hyps)
        try:
            m, _ = self.M.compute_score(refs, hyps)
            self.scores['METEOR'] += m
        except Exception as e:
            pass
        r, _ = self.R.compute_score(refs, hyps)
        c, _ = self.C.compute_score(refs, hyps)
        self.scores['BLEU_1'] += b1
        self.scores['BLEU_2'] += b2
        self.scores['BLEU_3'] += b3
        self.scores['BLEU_4'] += b4
        self.scores['ROUGE'] += r
        self.scores['CIDEr'] += c
        self.iterations += 1

    def reset_scores(self):
        """
        Resets each NLP metric in scores and iterations to 0
        """
        self.iterations = 0
        self.scores = {k: 0 for k, v in self.scores.items()}

    def get_scores(self):
        """
        Returns current scores value in Evaluator
        """
        i = self.iterations
        if i:
            scores = {k: (v / i) for k, v in self.scores.items()}
        else:
            scores = self.scores
        return scores

    # to-do: refactor outside of class
    def set_ckpt_scores(self, ckpt):
        """
        Sets checkpoint object's scores attribute to the current value of
        Evaluator's scores

        Parameters
        ----------
        ckpt: checkpoint object reference defined in ModelTrainer
        """
        try:
            scores = self.get_scores()
            for k, v in ckpt.scores.items():
                v.assign(scores[k])

        except AttributeError as e:
            print(repr(e))
