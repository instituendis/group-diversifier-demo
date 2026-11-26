from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .objective import ObjectiveModel, _cat_value, _multi_cluster_value


class SearchState:
    def __init__(self, model: ObjectiveModel, assignment: Sequence[int],
                 locks: Optional[Dict[int, int]] = None) -> None:
        if len(assignment) != model.n:
            raise ValueError(f"Assignment has {len(assignment)} entries for {model.n} students")
        self.model = model
        self.T = model.n_teams
        self.locks: Dict[int, int] = dict(locks or {})
        self.assignment: List[int] = [int(t) if t is not None else -1 for t in assignment]
        for i, t in enumerate(self.assignment):
            if not -1 <= t < self.T:
                raise ValueError(f"Student index {i} has invalid team {t}")
        for i, t in self.locks.items():
            if self.assignment[i] not in (-1, t):
                raise ValueError(f"Locked student index {i} is not in team {t}")
        self.fw = model.fairness_weight
        self._rebuild()

    def _rebuild(self) -> None:
        m = self.model
        T = self.T
        self.members: List[List[int]] = [[] for _ in range(T)]
        for i, t in enumerate(self.assignment):
            if t >= 0:
                self.members[t].append(i)
        self.sizes = [len(ms) for ms in self.members]
        self.cnt: List[List[int]] = []
        self.S2: List[List[int]] = []
        self.PN: List[List[int]] = []
        self.L: List[List[int]] = []
        self.M: List[List[int]] = []
        self.S1: List[List[float]] = []
        self.SQ: List[List[float]] = []
        self.GC: List[List[List[int]]] = []
        self.GU: List[List[float]] = []
        self.PTOT: List[float] = []
        self.MP: List[List[int]] = []
        self.BM: List[List[int]] = []
        self.BS: List[List[float]] = []
        self.BLO: List[List[float]] = []
        self.BHI: List[List[float]] = []
        self.SC: List[List[List[int]]] = []
        self.SH: List[List[int]] = []
        anchor = m.anchor
        self.away = (0 if anchor is None else
                     sum(1 for i, t in enumerate(self.assignment) if t >= 0 and 0 <= anchor[i] != t))
        for t in range(T):
            cnt, s2, pn, lsum, mm, s1, sq, gc, gu, ptot, ext, mp = m._aggregate(self.members[t])
            self.MP.append(mp)
            if ext is None:
                ext = ([], [], [], [], [], [])
            bm, bs, blo, bhi, sc, sh = ext
            self.BM.append(bm)
            self.BS.append(bs)
            self.BLO.append(blo)
            self.BHI.append(bhi)
            self.SC.append(sc)
            self.SH.append(sh)
            self.cnt.append(cnt)
            self.S2.append(s2)
            self.PN.append(pn)
            self.L.append(lsum)
            self.M.append(mm)
            self.S1.append(s1)
            self.SQ.append(sq)
            self.GC.append(gc)
            self.GU.append(gu)
            self.PTOT.append(ptot)
        self.PT: List[List[float]] = ([[0.0] * T for _ in range(m.n)] if m.has_pairs else [])
        if m.has_pairs:
            for i in range(m.n):
                row = self.PT[i]
                for j, w in m.neighbours[i]:
                    tj = self.assignment[j]
                    if tj >= 0:
                        row[tj] += w
        self.tv: List[List[float]] = [[] for _ in range(T)]
        self.coefs: List[List[float]] = [[] for _ in range(T)]
        self.P: List[float] = [0.0] * T
        for t in range(T):
            self._refresh_team(t)
        self._refresh_objective()

    def resync(self) -> None:
        self._rebuild()

    def _refresh_team(self, t: int) -> None:
        m = self.model
        s = self.sizes[t]
        ext = None
        if m.has_extra:
            if m.shared_rules:
                members = self.members[t]
                SC = self.SC[t]
                self.SH[t] = [m._shared_count(r, SC[r], members, s)
                              for r in range(len(m.shared_rules))]
            ext = (self.BM[t], self.BS[t], self.BLO[t], self.BHI[t], self.SC[t], self.SH[t])
        self.tv[t] = m._values(s, self.S2[t], self.PN[t], self.L[t], self.M[t], self.S1[t],
                               self.SQ[t], self.GU[t], self.PTOT[t], ext, self.MP[t])
        self.coefs[t] = [r.coef[s] if s < len(r.coef) else 0.0 for r in m.cat_rules]
        self.P[t] = math.fsum(self.tv[t])

    def _refresh_objective(self) -> None:
        P = self.P
        self.total = math.fsum(P)
        if self.T:
            order = sorted(range(self.T), key=lambda t: -P[t])
            self.top = order[:3]
            self.max_p = P[order[0]]
        else:
            self.top = []
            self.max_p = 0.0
        w = self.fw
        if w == 0.0:
            self.base_objective = self.total
        else:
            self.base_objective = (1.0 - w) * self.total + w * self.T * self.max_p
        self.objective = self.base_objective + self.away * self.model.anchor_weight

    @property
    def teams(self) -> List[List[int]]:
        return self.members

    @property
    def penalties(self) -> List[float]:
        return self.P

    def is_locked(self, i: int) -> bool:
        return i in self.locks

    def _side(self, t: int, x: int, y: int) -> float:
        m = self.model
        tv = self.tv[t]
        d = 0.0
        cx = m.cat_codes[x]
        cy = m.cat_codes[y]
        if cx != cy:
            cnt = self.cnt[t]
            S2 = self.S2[t]
            PN = self.PN[t]
            L = self.L[t]
            coefs = self.coefs[t]
            K = m.flat_k
            s = self.sizes[t]
            r = 0
            for a, b in zip(cx, cy):
                if a != b:
                    c = coefs[r]
                    if c != 0.0:
                        s2 = S2[r]
                        pn = PN[r]
                        lsum = L[r]
                        if a >= 0:
                            s2 += 1 - 2 * cnt[a]
                            pn -= 1
                            lsum -= K[a]
                        if b >= 0:
                            s2 += 2 * cnt[b] + 1
                            pn += 1
                            lsum += K[b]
                        rule = m.cat_rules[r]
                        g = rule.goal
                        if g == 0:
                            v = (s2 - lsum * (2.0 * s / rule.n_pool) + s * s * rule.sqn) * c
                        elif g == 1:
                            v = (s2 - pn) * 0.5 * c
                        else:
                            v = (pn * pn - s2) * 0.5 * c
                        d += v - tv[r]
                r += 1
        if m.n_cat_single != len(m.cat_rules):
            d += self._multi_side(t, x, y, tv)
        nx = m.num_values[x]
        if nx:
            ny = m.num_values[y]
            if nx != ny:
                M = self.M[t]
                S1 = self.S1[t]
                SQ = self.SQ[t]
                for r, (a, b) in enumerate(zip(nx, ny)):
                    if a == b:
                        continue
                    mm = M[r]
                    s1 = S1[r]
                    sq = SQ[r]
                    if a is not None:
                        mm -= 1
                        s1 -= a
                        sq -= a * a
                    if b is not None:
                        mm += 1
                        s1 += b
                        sq += b * b
                    rule = m.num_rules[r]
                    d += rule.value(mm, s1, sq) - tv[rule.term]
        gx = m.group_codes[x]
        if gx:
            gy = m.group_codes[y]
            if gx != gy:
                GC = self.GC[t]
                GU = self.GU[t]
                r = 0
                for a, b in zip(gx, gy):
                    if a != b:
                        grule = m.group_rules[r]
                        f = grule.table
                        cnt = GC[r]
                        u = GU[r]
                        for g in a:
                            if g not in b:
                                k = cnt[g]
                                u += f[k - 1] - f[k]
                        for h in b:
                            if h not in a:
                                k = cnt[h]
                                u += f[k + 1] - f[k]
                        d += u * grule.cost - tv[grule.term]
                    r += 1
        if m.has_extra:
            d += self._extra_side(t, x, y, tv)
        return d

    def _span_without(self, t: int, r: int, x: int) -> Tuple[float, float]:
        lo = math.inf
        hi = -math.inf
        bv = self.model.bound_values
        for k in self.members[t]:
            if k != x:
                v = bv[k][r]
                if v is not None:
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
        return lo, hi

    def _extra_side(self, t: int, x: int, y: int, tv: List[float]) -> float:
        m = self.model
        d = 0.0
        if m.bound_rules:
            bx = m.bound_values[x]
            by = m.bound_values[y]
            if bx != by:
                BM = self.BM[t]
                BS = self.BS[t]
                BLO = self.BLO[t]
                BHI = self.BHI[t]
                for r, brule in enumerate(m.bound_rules):
                    a = bx[r]
                    b = by[r]
                    if a == b:
                        continue
                    mm = BM[r]
                    s1 = BS[r]
                    lo = BLO[r]
                    hi = BHI[r]
                    if a is not None:
                        mm -= 1
                        s1 -= a
                    if b is not None:
                        mm += 1
                        s1 += b
                    if brule.max_range is not None:
                        if a is not None and (a <= lo or a >= hi):
                            lo, hi = self._span_without(t, r, x)
                        if b is not None:
                            if b < lo:
                                lo = b
                            if b > hi:
                                hi = b
                    d += brule.value(mm, s1, lo, hi) - tv[brule.term]
        if m.shared_rules:
            sx = m.shared_codes[x]
            sy = m.shared_codes[y]
            if sx != sy:
                s = self.sizes[t]
                SC = self.SC[t]
                SH = self.SH[t]
                for r, srule in enumerate(m.shared_rules):
                    a = sx[r]
                    b = sy[r]
                    if a == b:
                        continue
                    held = SC[r]
                    sh = SH[r]
                    for c in a:
                        if c not in b and held[c] == s:
                            sh -= 1
                    for c in b:
                        if c not in a and held[c] == s - 1:
                            sh += 1
                    d += srule.value(sh, s) - tv[srule.term]
        return d

    def _multi_side(self, t: int, x: int, y: int, tv: List[float]) -> float:
        m = self.model
        d = 0.0
        cnt = self.cnt[t]
        K = m.flat_k
        s = self.sizes[t]
        base = m.n_cat_single
        for rr, (a, b) in enumerate(zip(m.multi_codes[x], m.multi_codes[y])):
            if a == b:
                continue
            r = base + rr
            c = self.coefs[t][r]
            if c == 0.0:
                continue
            s2 = self.S2[t][r]
            pn = self.PN[t][r]
            lsum = self.L[t][r]
            for code in a:
                if code not in b:
                    s2 += 1 - 2 * cnt[code]
                    pn -= 1
                    lsum -= K[code]
            for code in b:
                if code not in a:
                    s2 += 2 * cnt[code] + 1
                    pn += 1
                    lsum += K[code]
            rule = m.cat_rules[r]
            if rule.goal == 2:
                mp = self.MP[t][rr] - (1 if a else 0) + (1 if b else 0)
                d += _multi_cluster_value(c, mp, pn, s2) - tv[r]
            else:
                d += _cat_value(rule.goal, c, s, s2, pn, lsum, rule.n_pool, rule.sqn) - tv[r]
        return d

    def _fair_delta(self, A: int, B: int, dA: float, dB: float) -> float:
        w = self.fw
        if w == 0.0:
            return dA + dB
        P = self.P
        new_max = P[A] + dA
        other = P[B] + dB
        if other > new_max:
            new_max = other
        for t in self.top:
            if t != A and t != B:
                if P[t] > new_max:
                    new_max = P[t]
                break
        return (1.0 - w) * (dA + dB) + w * self.T * (new_max - self.max_p)

    def delta_swap(self, i: int, j: int) -> float:
        a = self.assignment
        A = a[i]
        B = a[j]
        if A == B:
            return 0.0
        dA = self._side(A, i, j)
        dB = self._side(B, j, i)
        m = self.model
        if m.has_pairs:
            pti = self.PT[i]
            ptj = self.PT[j]
            w = m.pair_cost[i].get(j, 0.0)
            dA += ptj[A] - pti[A] - w
            dB += pti[B] - ptj[B] - w
        if m.anchor is None:
            return self._fair_delta(A, B, dA, dB)
        anchor = m.anchor
        ai, aj = anchor[i], anchor[j]
        away = ((0 <= ai != B) - (0 <= ai != A) + (0 <= aj != A) - (0 <= aj != B))
        return self._fair_delta(A, B, dA, dB) + away * m.anchor_weight

    def _penalty_with(self, t: int, out: Optional[int], inc: Optional[int]) -> float:
        m = self.model
        s = self.sizes[t] - (out is not None) + (inc is not None)
        s2 = list(self.S2[t])
        pn = list(self.PN[t])
        lsum = list(self.L[t])
        cnt = self.cnt[t]
        K = m.flat_k
        changes: Dict[int, int] = {}
        mp = list(self.MP[t])
        for student, sign in ((out, -1), (inc, 1)):
            if student is None:
                continue
            codes = list(m.cat_codes[student])
            for r, c in enumerate(codes):
                if c >= 0:
                    changes[c] = changes.get(c, 0) + sign
                    pn[r] += sign
                    lsum[r] += sign * K[c]
            for rr, mcodes in enumerate(m.multi_codes[student]):
                r = m.n_cat_single + rr
                if mcodes:
                    mp[rr] += sign
                for c in mcodes:
                    changes[c] = changes.get(c, 0) + sign
                    pn[r] += sign
                    lsum[r] += sign * K[c]
        if changes:
            owner = m.flat_owner
            for c, dc in changes.items():
                if dc:
                    k = cnt[c]
                    s2[owner[c]] += (k + dc) * (k + dc) - k * k
        mm = list(self.M[t])
        s1 = list(self.S1[t])
        sq = list(self.SQ[t])
        for student, sign in ((out, -1), (inc, 1)):
            if student is None:
                continue
            for r, v in enumerate(m.num_values[student]):
                if v is not None:
                    mm[r] += sign
                    s1[r] += sign * v
                    sq[r] += sign * v * v
        gu = list(self.GU[t])
        gcnt = self.GC[t]
        for r, grule in enumerate(m.group_rules):
            a = m.group_codes[out][r] if out is not None else ()
            b = m.group_codes[inc][r] if inc is not None else ()
            if a == b:
                continue
            f = grule.table
            delta: Dict[int, int] = {}
            for g in a:
                delta[g] = delta.get(g, 0) - 1
            for g in b:
                delta[g] = delta.get(g, 0) + 1
            u = gu[r]
            for g, dg in delta.items():
                if dg:
                    k = gcnt[r][g]
                    u += f[k + dg] - f[k]
            gu[r] = u
        ptot = self.PTOT[t]
        if m.has_pairs:
            if out is not None:
                ptot -= self.PT[out][t]
            if inc is not None:
                ptot += self.PT[inc][t]
                if out is not None:
                    ptot -= m.pair_cost[out].get(inc, 0.0)
        ext = self._extra_with(t, out, inc, s) if m.has_extra else None
        return math.fsum(m._values(s, s2, pn, lsum, mm, s1, sq, gu, ptot, ext, mp))

    def _extra_with(self, t: int, out: Optional[int], inc: Optional[int], s: int) -> Tuple:
        m = self.model
        bm = list(self.BM[t])
        bs = list(self.BS[t])
        blo = list(self.BLO[t])
        bhi = list(self.BHI[t])
        for r, brule in enumerate(m.bound_rules):
            a = m.bound_values[out][r] if out is not None else None
            b = m.bound_values[inc][r] if inc is not None else None
            if a is not None:
                bm[r] -= 1
                bs[r] -= a
            if b is not None:
                bm[r] += 1
                bs[r] += b
            if brule.max_range is not None:
                lo, hi = blo[r], bhi[r]
                if a is not None and (a <= lo or a >= hi):
                    lo, hi = self._span_without(t, r, out)
                if b is not None:
                    lo = min(lo, b)
                    hi = max(hi, b)
                blo[r], bhi[r] = lo, hi
        sh: List[int] = []
        if m.shared_rules:
            if inc is not None:
                ref: Optional[int] = inc
            else:
                ref = next((k for k in self.members[t] if k != out), None)
            for r in range(len(m.shared_rules)):
                if ref is None or s <= 0:
                    sh.append(0)
                    continue
                held = self.SC[t][r]
                a = m.shared_codes[out][r] if out is not None else ()
                b = m.shared_codes[inc][r] if inc is not None else ()
                count = 0
                for c in m.shared_codes[ref][r]:
                    if held[c] - (c in a) + (c in b) == s:
                        count += 1
                sh.append(count)
        return bm, bs, blo, bhi, None, sh

    def _objective_with(self, changed: Dict[int, float]) -> float:
        P = self.P
        total = self.total + sum(v - P[t] for t, v in changed.items())
        w = self.fw
        if w == 0.0:
            return total
        new_max = max(changed.values())
        for t in self.top:
            if t not in changed:
                if P[t] > new_max:
                    new_max = P[t]
                break
        else:
            if len(changed) < self.T:
                new_max = max(new_max, max(P[t] for t in range(self.T) if t not in changed))
        return (1.0 - w) * total + w * self.T * new_max

    def delta_move(self, i: int, t: int) -> float:
        A = self.assignment[i]
        if A == t:
            return 0.0
        changed = {t: self._penalty_with(t, None, i)}
        if A >= 0:
            changed[A] = self._penalty_with(A, i, None)
        d = self._objective_with(changed) - self.base_objective
        anchor = self.model.anchor
        if anchor is not None and anchor[i] >= 0:
            d += ((anchor[i] != t) - (A >= 0 and anchor[i] != A)) * self.model.anchor_weight
        return d

    def delta_add(self, i: int, t: int) -> float:
        return self.delta_move(i, t)

    def _add(self, t: int, i: int) -> None:
        self._shift(t, i, 1)

    def _remove(self, t: int, i: int) -> None:
        self._shift(t, i, -1)

    def _shift(self, t: int, i: int, sign: int) -> None:
        m = self.model
        cnt = self.cnt[t]
        K = m.flat_k
        for r, c in enumerate(m.cat_codes[i]):
            if c >= 0:
                k = cnt[c]
                self.S2[t][r] += sign * (2 * k + sign)
                cnt[c] = k + sign
                self.PN[t][r] += sign
                self.L[t][r] += sign * K[c]
        for rr, codes in enumerate(m.multi_codes[i]):
            r = m.n_cat_single + rr
            if codes:
                self.MP[t][rr] += sign
            for c in codes:
                k = cnt[c]
                self.S2[t][r] += sign * (2 * k + sign)
                cnt[c] = k + sign
                self.PN[t][r] += sign
                self.L[t][r] += sign * K[c]
        for r, v in enumerate(m.num_values[i]):
            if v is not None:
                self.M[t][r] += sign
                self.S1[t][r] += sign * v
                self.SQ[t][r] += sign * v * v
        for r, groups in enumerate(m.group_codes[i]):
            if groups:
                f = m.group_rules[r].table
                gc = self.GC[t][r]
                for g in groups:
                    k = gc[g]
                    self.GU[t][r] += f[k + sign] - f[k]
                    gc[g] = k + sign
        if m.has_pairs:
            self.PTOT[t] += sign * self.PT[i][t]
            PT = self.PT
            for j, w in m.neighbours[i]:
                PT[j][t] += sign * w
        self.sizes[t] += sign
        if sign > 0:
            self.members[t].append(i)
        else:
            self.members[t].remove(i)
        self.assignment[i] = t if sign > 0 else -1
        if m.has_extra:
            if sign > 0:
                self._extra_add(t, i)
            else:
                self._extra_remove(t, i)
        if m.anchor is not None and 0 <= m.anchor[i] != t:
            self.away += sign

    def _extra_remove(self, t: int, i: int) -> None:
        m = self.model
        if m.bound_rules:
            BM, BS, BLO, BHI = self.BM[t], self.BS[t], self.BLO[t], self.BHI[t]
            for r, v in enumerate(m.bound_values[i]):
                if v is not None:
                    BM[r] -= 1
                    BS[r] -= v
                    if m.bound_rules[r].max_range is not None and (v <= BLO[r] or v >= BHI[r]):
                        BLO[r], BHI[r] = self._span_without(t, r, -1)
        if m.shared_rules:
            SC = self.SC[t]
            for r, codes in enumerate(m.shared_codes[i]):
                held = SC[r]
                for c in codes:
                    held[c] -= 1

    def _extra_add(self, t: int, i: int) -> None:
        m = self.model
        if m.bound_rules:
            BM, BS, BLO, BHI = self.BM[t], self.BS[t], self.BLO[t], self.BHI[t]
            for r, v in enumerate(m.bound_values[i]):
                if v is not None:
                    BM[r] += 1
                    BS[r] += v
                    if v < BLO[r]:
                        BLO[r] = v
                    if v > BHI[r]:
                        BHI[r] = v
        if m.shared_rules:
            SC = self.SC[t]
            for r, codes in enumerate(m.shared_codes[i]):
                held = SC[r]
                for c in codes:
                    held[c] += 1

    def apply_swap(self, i: int, j: int) -> None:
        A = self.assignment[i]
        B = self.assignment[j]
        if A == B:
            return
        self._remove(A, i)
        self._remove(B, j)
        self._add(A, j)
        self._add(B, i)
        self._refresh_team(A)
        self._refresh_team(B)
        self._refresh_objective()

    def apply_move(self, i: int, t: int) -> None:
        A = self.assignment[i]
        if A == t:
            return
        if A >= 0:
            self._remove(A, i)
            self._refresh_team(A)
        self._add(t, i)
        self._refresh_team(t)
        self._refresh_objective()

    def apply_add(self, i: int, t: int) -> None:
        self.apply_move(i, t)

    def copy_assignment(self) -> List[int]:
        return list(self.assignment)
