package rlwe

import (
	"github.com/tuneinsight/lattigo/v4/ring"
	"github.com/tuneinsight/lattigo/v4/rlwe/ringqp"
	"github.com/tuneinsight/lattigo/v4/utils"
)

// GadgetProduct evaluates poly x Gadget -> RLWE where
//
// p0 = dot(decomp(cx) * gadget[0]) mod Q
// p1 = dot(decomp(cx) * gadget[1]) mod Q
//
// Expects the flag IsNTT of ct to correctly reflect the domain of cx.
func (eval *Evaluator) GadgetProduct(levelQ int, cx *ring.Poly, gadgetCt GadgetCiphertext, ct *Ciphertext) {

	levelQ = utils.MinInt(levelQ, gadgetCt.LevelQ())
	levelP := gadgetCt.LevelP()

	ctTmp := CiphertextQP{}
	ctTmp.Value = [2]ringqp.Poly{{Q: ct.Value[0], P: eval.BuffQP[1].P}, {Q: ct.Value[1], P: eval.BuffQP[2].P}}
	ctTmp.IsNTT = ct.IsNTT

	if levelP > 0 {
		eval.GadgetProductNoModDown(levelQ, cx, gadgetCt, ctTmp)
	} else {
		eval.GadgetProductSinglePAndBitDecompNoModDown(levelQ, cx, gadgetCt, ctTmp)
	}

	// now := time.Now()
	if ct.IsNTT && levelP != -1 {
		eval.BasisExtender.ModDownQPtoQNTT(levelQ, levelP, ct.Value[0], ctTmp.Value[0].P, ct.Value[0])
		eval.BasisExtender.ModDownQPtoQNTT(levelQ, levelP, ct.Value[1], ctTmp.Value[1].P, ct.Value[1])
	} else if !ct.IsNTT {

		eval.params.RingQ().InvNTTLazyLvl(levelQ, ct.Value[0], ct.Value[0])
		eval.params.RingQ().InvNTTLazyLvl(levelQ, ct.Value[1], ct.Value[1])

		if levelP != -1 {
			eval.params.RingP().InvNTTLazyLvl(levelP, ctTmp.Value[0].P, ctTmp.Value[0].P)
			eval.params.RingP().InvNTTLazyLvl(levelP, ctTmp.Value[1].P, ctTmp.Value[1].P)

			eval.BasisExtender.ModDownQPtoQ(levelQ, levelP, ct.Value[0], ctTmp.Value[0].P, ct.Value[0])
			eval.BasisExtender.ModDownQPtoQ(levelQ, levelP, ct.Value[1], ctTmp.Value[1].P, ct.Value[1])
		}
	}
	// fmt.Printf("ModDown using time %s\n", time.Since(now))
}

// GadgetProductNoModDown applies the gadget prodcut to the polynomial cx:
//
// ct.Value[0] = dot(decomp(cx) * gadget[0]) mod QP (encrypted input is multiplied by P factor)
// ct.Value[1] = dot(decomp(cx) * gadget[1]) mod QP (encrypted input is multiplied by P factor)
//
// Expects the flag IsNTT of ct to correctly reflect the domain of cx.
func (eval *Evaluator) GadgetProductNoModDown(levelQ int, cx *ring.Poly, gadgetCt GadgetCiphertext, ct CiphertextQP) {

	ringQ := eval.params.RingQ()
	ringP := eval.params.RingP()
	ringQP := eval.params.RingQP()

	c2QP := eval.BuffQP[0]

	var cxNTT, cxInvNTT *ring.Poly
	if ct.IsNTT {
		cxNTT = cx
		cxInvNTT = eval.BuffInvNTT
		ringQ.InvNTTLvl(levelQ, cxNTT, cxInvNTT)
	} else {
		cxNTT = eval.BuffInvNTT
		cxInvNTT = cx
		ringQ.NTTLvl(levelQ, cxInvNTT, cxNTT)
	}

	levelP := gadgetCt.LevelP()

	decompRNS := eval.params.DecompRNS(levelQ, levelP)

	QiOverF := eval.params.QiOverflowMargin(levelQ) >> 1
	PiOverF := eval.params.PiOverflowMargin(levelP) >> 1

	el := gadgetCt.Value

	// Key switching with CRT decomposition for the Qi
	var reduce int
	for i := 0; i < decompRNS; i++ {

		eval.DecomposeSingleNTT(levelQ, levelP, levelP+1, i, cxNTT, cxInvNTT, c2QP.Q, c2QP.P)

		if i == 0 {
			ringQP.MulCoeffsMontgomeryConstantLvl(levelQ, levelP, el[i][0].Value[0], c2QP, ct.Value[0])
			ringQP.MulCoeffsMontgomeryConstantLvl(levelQ, levelP, el[i][0].Value[1], c2QP, ct.Value[1])
		} else {
			ringQP.MulCoeffsMontgomeryConstantAndAddNoModLvl(levelQ, levelP, el[i][0].Value[0], c2QP, ct.Value[0])
			ringQP.MulCoeffsMontgomeryConstantAndAddNoModLvl(levelQ, levelP, el[i][0].Value[1], c2QP, ct.Value[1])
		}

		if reduce%QiOverF == QiOverF-1 {
			ringQ.ReduceLvl(levelQ, ct.Value[0].Q, ct.Value[0].Q)
			ringQ.ReduceLvl(levelQ, ct.Value[1].Q, ct.Value[1].Q)
		}

		if reduce%PiOverF == PiOverF-1 {
			ringP.ReduceLvl(levelP, ct.Value[0].P, ct.Value[0].P)
			ringP.ReduceLvl(levelP, ct.Value[1].P, ct.Value[1].P)
		}

		reduce++
	}

	if reduce%QiOverF != 0 {
		ringQ.ReduceLvl(levelQ, ct.Value[0].Q, ct.Value[0].Q)
		ringQ.ReduceLvl(levelQ, ct.Value[1].Q, ct.Value[1].Q)
	}

	if reduce%PiOverF != 0 {
		ringP.ReduceLvl(levelP, ct.Value[0].P, ct.Value[0].P)
		ringP.ReduceLvl(levelP, ct.Value[1].P, ct.Value[1].P)
	}
}

// GadgetProductSinglePAndBitDecompNoModDown applies the key-switch to the polynomial cx:
//
// ct.Value[0] = dot(decomp(cx) * evakey[0]) mod QP (encrypted input is multiplied by P factor)
// ct.Value[1] = dot(decomp(cx) * evakey[1]) mod QP (encrypted input is multiplied by P factor)
//
// Expects the flag IsNTT of ct to correctly reflect the domain of cx.
func (eval *Evaluator) GadgetProductSinglePAndBitDecompNoModDown(levelQ int, cx *ring.Poly, gadgetCt GadgetCiphertext, ct CiphertextQP) {

	ringQ := eval.params.RingQ()
	ringP := eval.params.RingP()

	var cxInvNTT *ring.Poly
	if ct.IsNTT {
		cxInvNTT = eval.BuffInvNTT
		ringQ.InvNTTLvl(levelQ, cx, cxInvNTT)
	} else {
		cxInvNTT = cx
	}

	levelP := gadgetCt.LevelP()

	decompRNS := eval.params.DecompRNS(levelQ, levelP)
	decompPw2 := eval.params.DecompPw2(levelQ, levelP)

	pw2 := eval.params.pow2Base

	mask := uint64(((1 << pw2) - 1))

	if mask == 0 {
		mask = 0xFFFFFFFFFFFFFFFF
	}

	cw := eval.BuffQP[0].Q.Coeffs[0]
	cwNTT := eval.BuffBitDecomp

	QiOverF := eval.params.QiOverflowMargin(levelQ) >> 1
	PiOverF := eval.params.PiOverflowMargin(levelP) >> 1

	el := gadgetCt.Value

	// Key switching with CRT decomposition for the Qi
	var reduce int
	for i := 0; i < decompRNS; i++ {
		for j := 0; j < decompPw2; j++ {

			ring.MaskVec(cxInvNTT.Coeffs[i], cw, j*pw2, mask)

			if i == 0 && j == 0 {
				for u := 0; u < levelQ+1; u++ {
					ringQ.NTTSingleLazy(u, cw, cwNTT)
					ring.MulCoeffsMontgomeryConstantVec(el[i][j].Value[0].Q.Coeffs[u], cwNTT, ct.Value[0].Q.Coeffs[u], ringQ.Modulus[u], ringQ.MredParams[u])
					ring.MulCoeffsMontgomeryConstantVec(el[i][j].Value[1].Q.Coeffs[u], cwNTT, ct.Value[1].Q.Coeffs[u], ringQ.Modulus[u], ringQ.MredParams[u])
				}

				for u := 0; u < levelP+1; u++ {
					ringP.NTTSingleLazy(u, cw, cwNTT)
					ring.MulCoeffsMontgomeryConstantVec(el[i][j].Value[0].P.Coeffs[u], cwNTT, ct.Value[0].P.Coeffs[u], ringP.Modulus[u], ringP.MredParams[u])
					ring.MulCoeffsMontgomeryConstantVec(el[i][j].Value[1].P.Coeffs[u], cwNTT, ct.Value[1].P.Coeffs[u], ringP.Modulus[u], ringP.MredParams[u])
				}
			} else {
				for u := 0; u < levelQ+1; u++ {
					ringQ.NTTSingleLazy(u, cw, cwNTT)
					ring.MulCoeffsMontgomeryConstantAndAddNoModVec(el[i][j].Value[0].Q.Coeffs[u], cwNTT, ct.Value[0].Q.Coeffs[u], ringQ.Modulus[u], ringQ.MredParams[u])
					ring.MulCoeffsMontgomeryConstantAndAddNoModVec(el[i][j].Value[1].Q.Coeffs[u], cwNTT, ct.Value[1].Q.Coeffs[u], ringQ.Modulus[u], ringQ.MredParams[u])
				}

				for u := 0; u < levelP+1; u++ {
					ringP.NTTSingleLazy(u, cw, cwNTT)
					ring.MulCoeffsMontgomeryConstantAndAddNoModVec(el[i][j].Value[0].P.Coeffs[u], cwNTT, ct.Value[0].P.Coeffs[u], ringP.Modulus[u], ringP.MredParams[u])
					ring.MulCoeffsMontgomeryConstantAndAddNoModVec(el[i][j].Value[1].P.Coeffs[u], cwNTT, ct.Value[1].P.Coeffs[u], ringP.Modulus[u], ringP.MredParams[u])
				}
			}

			if reduce%QiOverF == QiOverF-1 {
				ringQ.ReduceLvl(levelQ, ct.Value[0].Q, ct.Value[0].Q)
				ringQ.ReduceLvl(levelQ, ct.Value[1].Q, ct.Value[1].Q)
			}

			if reduce%PiOverF == PiOverF-1 {
				ringP.ReduceLvl(levelP, ct.Value[0].P, ct.Value[0].P)
				ringP.ReduceLvl(levelP, ct.Value[1].P, ct.Value[1].P)
			}

			reduce++
		}
	}

	if reduce%QiOverF != 0 {
		ringQ.ReduceLvl(levelQ, ct.Value[0].Q, ct.Value[0].Q)
		ringQ.ReduceLvl(levelQ, ct.Value[1].Q, ct.Value[1].Q)
	}

	if reduce%PiOverF != 0 {
		ringP.ReduceLvl(levelP, ct.Value[0].P, ct.Value[0].P)
		ringP.ReduceLvl(levelP, ct.Value[1].P, ct.Value[1].P)
	}
}

func (eval *Evaluator) GadgetProductAndRescale(levelQ int, c0 *ring.Poly, cx *ring.Poly, gadgetCt GadgetCiphertext, ct *Ciphertext) {

	levelQ = utils.MinInt(levelQ, gadgetCt.LevelQ())
	levelP := gadgetCt.LevelP()

	ringQ := eval.params.RingQ()

	ctTmp := CiphertextQP{}
	ctTmp.Value = [2]ringqp.Poly{{Q: ct.Value[0], P: eval.BuffQP[1].P}, {Q: ct.Value[1], P: eval.BuffQP[2].P}}
	ctTmp.IsNTT = ct.IsNTT

	if levelP > 0 {
		eval.GadgetProductNoModDown(levelQ, cx, gadgetCt, ctTmp)
	} else {
		eval.GadgetProductSinglePAndBitDecompNoModDown(levelQ, cx, gadgetCt, ctTmp)
	}

	if ct.IsNTT && levelP != -1 {
		//eval.BasisExtender.ModDownQPtoQNTT(levelQ, levelP, ct.Value[0], ctTmp.Value[0].P, ct.Value[0])
		//eval.BasisExtender.ModDownQPtoQNTT(levelQ, levelP, ct.Value[1], ctTmp.Value[1].P, ct.Value[1])

		ringQ.MulScalarBigintLvl(levelQ, c0, eval.params.RingP().ModulusAtLevel[levelP], eval.BuffQP[4].Q)
		ringQ.AddLvl(levelQ, ctTmp.Value[0].Q, eval.BuffQP[4].Q, ctTmp.Value[0].Q)

		N := eval.params.N()
		copy(eval.BuffPq.Buff, ctTmp.Value[0].P.Buff) // fill p1Pq with p1P in the lower space.
		if len(eval.BuffPq.Coeffs[levelP+1:]) != 1 {
			panic("len(eval.BuffPq.Coeffs[levelP+1:]) != 1")
		}
		copy(eval.BuffPq.Buff[(levelP+1)*N:], ctTmp.Value[0].Q.Buff[(levelQ)*N:]) // fill p1Pq with p1Q's levelQ-th component at the higher space

		eval.BasisExtender.ModDownQPtoQlm1NTT(levelQ, levelP, ctTmp.Value[0].Q, eval.BuffPq, ct.Value[0])

		copy(eval.BuffPq.Buff, ctTmp.Value[1].P.Buff) // fill p1Pq with p1P in the lower space.
		if len(eval.BuffPq.Coeffs[levelP+1:]) != 1 {
			panic("len(eval.BuffPq.Buff[levelP+1:]) != 1")
		}
		copy(eval.BuffPq.Buff[(levelP+1)*N:], ctTmp.Value[1].Q.Buff[(levelQ)*N:]) // fill p1Pq with p1Q's levelQ-th component at the higher space

		eval.BasisExtender.ModDownQPtoQlm1NTT(levelQ, levelP, ctTmp.Value[1].Q, eval.BuffPq, ct.Value[1])

	} else if !ct.IsNTT {
		panic("ct should be in NTT domain")
	}
}
