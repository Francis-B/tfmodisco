# core.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar

import pickle

import numpy as np
import scipy.sparse

from . import affinitymat
from . import cluster
from . import util

from collections import OrderedDict


class TrackSet(object):
    def __init__(self, one_hot, contrib_scores, hypothetical_contribs):
        self.one_hot = one_hot
        self.contrib_scores = contrib_scores
        self.hypothetical_contribs = hypothetical_contribs

    def create_seqlets(self, seqlets):
        for seqlet in seqlets:
            idx = seqlet.example_idx
            s, e = seqlet.start, seqlet.end

            if seqlet.is_revcomp:
                seqlet.sequence = self.one_hot[idx][s:e][::-1, ::-1]
                seqlet.contrib_scores = self.contrib_scores[idx][s:e][::-1, ::-1]
                seqlet.hypothetical_contribs = self.hypothetical_contribs[idx][s:e][
                    ::-1, ::-1
                ]
            else:
                seqlet.sequence = self.one_hot[idx][s:e]
                seqlet.contrib_scores = self.contrib_scores[idx][s:e]
                seqlet.hypothetical_contribs = self.hypothetical_contribs[idx][s:e]

        return seqlets

    def get_track_length(self, example_idx):
        """
        Get the length of the track at the given index. This method was added to account
        for inputs with different lengths.

        Parameters
        ---------
        example_idx : int
            Index of the track for which the length will be returned.

        Returns
        -------
        int
            Length of the track.
        """
        return self.one_hot[example_idx].shape[0]


class Seqlet(object):
    def __init__(self, example_idx, start, end, is_revcomp):
        self.example_idx = example_idx
        self.start = start
        self.end = end
        self.is_revcomp = is_revcomp

        self.sequence = None
        self.contrib_scores = None
        self.hypothetical_contribs = None

        super(Seqlet, self).__init__()

    def __str__(self):
        return (
            "example:"
            + str(self.example_idx)
            + ",start:"
            + str(self.start)
            + ",end:"
            + str(self.end)
            + ",rc:"
            + str(self.is_revcomp)
        )

    def __len__(self):
        return self.end - self.start

    @property
    def string(self):
        return str(self.example_idx) + "_" + str(self.start) + "_" + str(self.end)

    def revcomp(self):
        new_seqlet = Seqlet(
            example_idx=self.example_idx,
            start=self.start,
            end=self.end,
            is_revcomp=not self.is_revcomp,
        )

        if self.sequence is None:
            raise ValueError("Seqlet.sequence was not define yet")
        if self.contrib_scores is None:
            raise ValueError("Seqlet.contrib_scores was not define yet")
        if self.hypothetical_contribs is None:
            raise ValueError("Seqlet.hypothetical_contribs was not define yet")

        new_seqlet.sequence = self.sequence[::-1, ::-1]
        new_seqlet.contrib_scores = self.contrib_scores[::-1, ::-1]
        new_seqlet.hypothetical_contribs = self.hypothetical_contribs[::-1, ::-1]
        return new_seqlet

    def shift(self, shift_amt):
        return Seqlet(
            example_idx=self.example_idx,
            start=self.start + shift_amt,
            end=self.end + shift_amt,
            is_revcomp=self.is_revcomp,
        )

    def trim(self, start_idx, end_idx):

        if self.sequence is None:
            raise ValueError("Seqlet.sequence was not define yet")
        if self.contrib_scores is None:
            raise ValueError("Seqlet.contrib_scores was not define yet")
        if self.hypothetical_contribs is None:
            raise ValueError("Seqlet.hypothetical_contribs was not define yet")

        # Correct idx if they go beyond sequence
        s, e = start_idx, end_idx
        if e > self.sequence.shape[0]:
            desired_length = e - s
            e = self.sequence.shape[0]
            s = e - desired_length

        if not self.is_revcomp:
            new_start = self.start + s
            new_end = self.start + e
        else:
            new_start = self.end - e
            new_end = self.end - s

        new_seqlet = Seqlet(
            example_idx=self.example_idx,
            start=new_start,
            end=new_end,
            is_revcomp=self.is_revcomp,
        )

        new_seqlet.sequence = self.sequence[s:e]
        new_seqlet.contrib_scores = self.contrib_scores[s:e]
        new_seqlet.hypothetical_contribs = self.hypothetical_contribs[s:e]

        return new_seqlet


class SeqletSet:
    def __init__(self, seqlets):
        self.seqlets = []
        self.unique_seqlets = {}
        self.length = max([len(seqlet.sequence) for seqlet in seqlets])

        self._sequence_sum = np.zeros((self.length, 4), dtype="float")
        self._contrib_sum = np.zeros((self.length, 4), dtype="float")
        self._hypothetical_sum = np.zeros((self.length, 4), dtype="float")

        self.sequence = np.zeros((self.length, 4), dtype="float")
        self.contrib_scores = np.zeros((self.length, 4), dtype="float")
        self.hypothetical_contribs = np.zeros((self.length, 4), dtype="float")

        self.per_position_counts = np.zeros((self.length,))

        for seqlet in seqlets:
            if seqlet.string not in self.unique_seqlets:
                self._add_seqlet(seqlet=seqlet)

        self.subclusters = None
        self.subcluster_to_subpattern = None

    def compute_subpatterns(self, perplexity, n_seeds, n_iterations=-1):
        """
        Parameters
        ----------
        perplexity : int
            Perplexity value to pass to the Leiden clustering function.
        n_seeds : int
            Random seed.
        n_iterations : int
            Number of iteration for the Leiden clustering. Set to -1 to let the
            leiden function decide of the number of iterations.
        """
        # this method assumes all the seqlets have been expanded so they
        # all start at 0
        X = util.get_2d_data_from_patterns(self.seqlets)[0]
        X = X.reshape(len(X), -1)

        n = len(X)
        n_neighb = min(int(perplexity * 3 + 2), len(X))

        affmat_nn, seqlet_neighbors = affinitymat.pairwise_jaccard(X, n_neighb)

        distmat_nn = np.log((1.0 / (0.5 * np.maximum(affmat_nn, 0.0000001))) - 1)
        distmat_nn = np.maximum(distmat_nn, 0.0)  # eliminate tiny neg floats

        distmat_sp = scipy.sparse.coo_matrix(
            (
                np.concatenate(distmat_nn, axis=0),
                (
                    np.array(
                        [
                            i
                            for i in range(len(seqlet_neighbors))
                            for j in seqlet_neighbors[i]
                        ]
                    ).astype("int"),
                    np.concatenate(seqlet_neighbors, axis=0),
                ),
            ),
            shape=(len(distmat_nn), len(distmat_nn)),
        ).tocsr()

        distmat_sp.sort_indices()

        # do density adaptation
        sp_density_adapted_affmat = affinitymat.NNTsneConditionalProbs(
            perplexity=perplexity
        )(affmat_nn, seqlet_neighbors)

        sp_density_adapted_affmat += sp_density_adapted_affmat.T
        sp_density_adapted_affmat /= np.sum(sp_density_adapted_affmat.data)

        # Do Leiden clustering
        self.subclusters = cluster.LeidenClusterParallel(
            sp_density_adapted_affmat,
            n_seeds=n_seeds,
            n_leiden_iterations=n_iterations,
            verbose=False,
        )

        # this method assumes all the seqlets have been expanded so they
        # all start at 0
        subcluster_to_seqletsandalignments = OrderedDict()
        for seqlet, subcluster in zip(self.seqlets, self.subclusters):
            if subcluster not in subcluster_to_seqletsandalignments:
                subcluster_to_seqletsandalignments[subcluster] = []

            subcluster_to_seqletsandalignments[subcluster].append(seqlet)

        subcluster_to_subpattern = OrderedDict(
            [
                (subcluster, SeqletSet(seqletsandalignments))
                for subcluster, seqletsandalignments in subcluster_to_seqletsandalignments.items()
            ]
        )

        # resort subcluster_to_subpattern so that the subclusters with the
        # most seqlets come first
        self.subcluster_to_subpattern = OrderedDict(
            sorted(subcluster_to_subpattern.items(), key=lambda x: -len(x[1].seqlets))
        )

    def copy(self):
        return SeqletSet(seqlets=[seqlet for seqlet in self.seqlets])

    def trim_to_support(self, min_frac, min_num, min_length):
        """
        Trim the border of the pattern that are supported by less than provided
        proportion or number of seqlet. Will use the minimum value between
        min_frac and min_num.

        Parameters
        ----------
        min_frac : float
            Minimum proportion of supporting seqlets needed to not remove a base.
        min_num : int
            Minimum number of supporting seqlets needed to not remove a base.
        min_length : int
            Minimum length to trim to.

        Returns
        -------
        SeqletSet
            Trimmed SeqletSet.
        """

        # If not all position are covered by the same number of seqlet
        # This happens when then is a really strong motif that center all
        # seqlet on the same position (e.g. start codon)
        if len(set(self.per_position_counts)) != 1:
            values = self.per_position_counts
            max_support = max(values)
            num = min(min_num, max_support * min_frac)

        else:
            values = self.per_position_counts * self.contrib_scores.sum(axis=1)
            max_support = max(values)
            num = max_support * min_frac

        left_idx = 0
        right_idx = len(values) - 1

        # Trim the border of the seqlet until we get the minimum length or the
        # minimal number of support.
        while right_idx - left_idx > min_length:
            if values[right_idx] >= num and values[left_idx] >= num:
                break
            if values[right_idx] < values[left_idx]:
                right_idx -= 1
            else:
                left_idx += 1

        return self.trim_to_idx(start_idx=left_idx, end_idx=right_idx)

    def trim_to_idx(self, start_idx, end_idx):
        """
        Trim the seqlet at the given indexes.

        Parameters
        ----------
        start_idx : int
            Index at which the new seqlets will start.
        end_idx : int
            Index at which the new seqlets will end.

        Returns
        -------
        SeqletSet
            Trimmed SeqletSet.
        """
        new_seqlets = []
        for seqlet in self.seqlets:
            new_seqlet = seqlet.trim(start_idx=start_idx, end_idx=end_idx)
            new_seqlets.append(new_seqlet)
        return SeqletSet(seqlets=new_seqlets)

    def _add_seqlet(self, seqlet):
        n = seqlet.sequence.shape[0]

        self.seqlets.append(seqlet)
        self.unique_seqlets[seqlet.string] = seqlet
        self.per_position_counts[:n] += 1.0

        ppc = self.per_position_counts[:, None]
        ppc = ppc + 1e-7 * (ppc == 0)

        self._sequence_sum[:n] += seqlet.sequence
        self._contrib_sum[:n] += seqlet.contrib_scores
        self._hypothetical_sum[:n] += seqlet.hypothetical_contribs

        self.sequence = self._sequence_sum / ppc
        self.contrib_scores = self._contrib_sum / ppc
        self.hypothetical_contribs = self._hypothetical_sum / ppc

    def __len__(self):
        return self.length

    def save_seqlets(self, filename):
        bases = np.array(["A", "C", "G", "T"])

        with open(filename, "w") as outfile:
            for seqlet in self.seqlets:
                sequence = "".join(bases[np.argmax(seqlet.sequence, axis=-1)])
                example_index = seqlet.example_idx
                start, end = seqlet.start, seqlet.end
                outfile.write(">example%d:%d-%d\n" % (example_index, start, end))
                outfile.write(sequence + "\n")
