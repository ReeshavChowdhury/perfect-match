package ckks

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"sort"
	"time"

	"github.com/tuneinsight/lattigo/v4/ring"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v4/rlwe/ringqp"
	"github.com/tuneinsight/lattigo/v4/utils"
)

// Evaluator is an interface implementing the methods to conduct homomorphic operations between ciphertext and/or plaintexts.
type Evaluator interface {
	// ========================
	// === Basic Arithmetic ===
	// ========================

	// Addition
	Add(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext)
	AddNew(ctIn *rlwe.Ciphertext, op1 rlwe.Operand) (ctOut *rlwe.Ciphertext)

	// Subtraction
	Sub(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext)
	SubNew(ctIn *rlwe.Ciphertext, op1 rlwe.Operand) (ctOut *rlwe.Ciphertext)

	// Negation
	Neg(ctIn *rlwe.Ciphertext, ctOut *rlwe.Ciphertext)
	NegNew(ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext)

	// Constant Addition
	AddConstNew(ctIn *rlwe.Ciphertext, constant interface{}) (ctOut *rlwe.Ciphertext)
	AddConst(ctIn *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext)

	// Constant Multiplication
	MultByConstNew(ctIn *rlwe.Ciphertext, constant interface{}) (ctOut *rlwe.Ciphertext)
	MultByConst(ctIn *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext)
	MultByGaussianInteger(ctIn *rlwe.Ciphertext, cReal, cImag interface{}, ctOut *rlwe.Ciphertext)

	// Constant Multiplication with Addition
	MultByConstAndAdd(ctIn *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext)
	MultByGaussianIntegerAndAdd(ctIn *rlwe.Ciphertext, cReal, cImag interface{}, ctOut *rlwe.Ciphertext)

	// Multiplication by the imaginary unit
	MultByiNew(ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext)
	MultByi(ctIn *rlwe.Ciphertext, ctOut *rlwe.Ciphertext)
	DivByiNew(ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext)
	DivByi(ctIn *rlwe.Ciphertext, ctOut *rlwe.Ciphertext)

	// Conjugation
	ConjugateNew(ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext)
	Conjugate(ctIn *rlwe.Ciphertext, ctOut *rlwe.Ciphertext)

	// Multiplication
	Mul(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext)
	MulNew(ctIn *rlwe.Ciphertext, op1 rlwe.Operand) (ctOut *rlwe.Ciphertext)
	MulRelin(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext)
	MulRelinNew(ctIn *rlwe.Ciphertext, op1 rlwe.Operand) (ctOut *rlwe.Ciphertext)

	MulAndAdd(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext)
	MulRelinAndAdd(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext)
	MulAndSum(ctIn []*rlwe.Ciphertext, ptIn []*rlwe.Plaintext, ctOut *rlwe.Ciphertext)

	// Slot Rotations
	RotateNew(ctIn *rlwe.Ciphertext, k int) (ctOut *rlwe.Ciphertext)
	Rotate(ctIn *rlwe.Ciphertext, k int, ctOut *rlwe.Ciphertext)
	RotateHoistedNew(ctIn *rlwe.Ciphertext, rotations []int) (ctOut map[int]*rlwe.Ciphertext)
	RotateHoisted(ctIn *rlwe.Ciphertext, rotations []int, ctOut map[int]*rlwe.Ciphertext)
	RotateHoistedNoModDownNew(level int, rotations []int, c0 *ring.Poly, c2DecompQP []ringqp.Poly) (cOut map[int]rlwe.CiphertextQP)
	RotateHoistedModDownAndRescaleOne(ctIn *rlwe.Ciphertext, rotation int, ctOut *rlwe.Ciphertext)
	RotateHoistedAndRescaleOne(ctIn *rlwe.Ciphertext, rotation int, ctOut *rlwe.Ciphertext)

	// ===========================
	// === Advanced Arithmetic ===
	// ===========================

	// Polynomial evaluation
	EvaluatePoly(input interface{}, pol *Polynomial, targetScale rlwe.Scale) (ctOut *rlwe.Ciphertext, err error)
	EvaluatePolyVector(input interface{}, pols []*Polynomial, encoder Encoder, slotIndex map[int][]int, targetScale rlwe.Scale) (ctOut *rlwe.Ciphertext, err error)

	// Inversion
	InverseNew(ctIn *rlwe.Ciphertext, steps int) (ctOut *rlwe.Ciphertext, err error)

	// Linear Transformations
	LinearTransform4ArithmeticSeqNew(ctIn *rlwe.Ciphertext, linearTransform interface{}, shareInnerLoop ...bool) (ctOut []*rlwe.Ciphertext)
	LinearTransformNew(ctIn *rlwe.Ciphertext, linearTransform interface{}) (ctOut []*rlwe.Ciphertext)
	LinearTransform(ctIn *rlwe.Ciphertext, linearTransform interface{}, ctOut []*rlwe.Ciphertext)
	MultiplyByDiagMatrix(ctIn *rlwe.Ciphertext, matrix LinearTransform, c2DecompQP []ringqp.Poly, ctOut *rlwe.Ciphertext)
	MultiplyByDiagMatrixBSGS(ctIn *rlwe.Ciphertext, matrix LinearTransform, c2DecompQP []ringqp.Poly, ctOut *rlwe.Ciphertext)
	ColShiftRestricted(ctIn *rlwe.Ciphertext, LTs []LinearTransform, AvailableRots int, dimension int) (ctOut []*rlwe.Ciphertext)
	ColShiftRestricted_batch(ctIn *rlwe.Ciphertext, LTs []LinearTransform, AvailableRots int, dimension int, batchsize int) (ctOut []*rlwe.Ciphertext)
	LinearTransformNew_flexkey(ctIn *rlwe.Ciphertext, linearTransform interface{}, Rots_submap map[int][]int, sk *rlwe.SecretKey) (ctOut []*rlwe.Ciphertext)
	MultiplyByDiagMatrixBSGS_flexkey(ctIn *rlwe.Ciphertext, matrix LinearTransform, PoolDecompQP []ringqp.Poly, ctOut *rlwe.Ciphertext, Rots_submap map[int][]int, sk *rlwe.SecretKey)

	// Inner sum
	InnerSum(ctIn *rlwe.Ciphertext, batch, n int, ctOut *rlwe.Ciphertext)
	Average(ctIn *rlwe.Ciphertext, batch int, ctOut *rlwe.Ciphertext)

	// Replication (inverse of Inner sum)
	Replicate(ctIn *rlwe.Ciphertext, batch, n int, ctOut *rlwe.Ciphertext)

	// Trace
	Trace(ctIn *rlwe.Ciphertext, logSlots int, ctOut *rlwe.Ciphertext)
	TraceNew(ctIn *rlwe.Ciphertext, logSlots int) (ctOut *rlwe.Ciphertext)

	// Permutation
	MultiGroupNetworkTopDownNew(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext)
	MultiGroupNetworkTopDownNewV2(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext)
	PrepareNexAndPre(MMaskpt [][][][]map[string]*rlwe.Plaintext) (pre, nex [][]map[int]*rlwe.Ciphertext, nex_cnt [][]map[int]int)
	MultiGroupNetworkTopDownNewV3(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, pre, nex [][]map[int]*rlwe.Ciphertext, nex_cnt [][]map[int]int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext)
	MultiGroupNetworkTopDownNewV3_tick(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, pre, nex [][]map[int]*rlwe.Ciphertext, nex_cnt [][]map[int]int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext, seg []time.Duration)
	MultiGroupNetworkCollapseUpper(ctIn *rlwe.Ciphertext, collapsed collapsedlevelsInfo, sign int, n int, sk *rlwe.SecretKey, nex [][]map[int]*rlwe.Ciphertext)
	TreebasedHoistedRotation(ctIn *rlwe.Ciphertext, level int, unitstep int, DiffRots []int, rotated int, resultmap map[int]rlwe.CiphertextQP, sk *rlwe.SecretKey)
	MultiGroupNetworkCollapseBottomNew(collapsed collapsedlevelsInfoB, sign int, n int, sk *rlwe.SecretKey, nex [][]map[int]*rlwe.Ciphertext) (ctOut *rlwe.Ciphertext)
	MultiGroupNetworkTopDownNewV4LvlCollapse(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, pre, nex [][]map[int]*rlwe.Ciphertext, nex_cnt [][]map[int]int, UpCInfo collapsedlevelsInfo, BotCInfo collapsedlevelsInfoB, ecd_levelEachGroup []map[int]int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext)
	//MultiGroupNetworkTopDownNewV4LvlCollapse(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, pre, nex [][]map[int]*rlwe.Ciphertext, nex_cnt [][]map[int]int, UpCInfo collapsedlevelsInfo, BotCInfo collapsedlevelsInfoB, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext)

	// =============================
	// === Ciphertext Management ===
	// =============================

	// Key-Switching
	SwitchKeysNew(ctIn *rlwe.Ciphertext, switchingKey *rlwe.SwitchingKey) (ctOut *rlwe.Ciphertext)
	SwitchKeys(ctIn *rlwe.Ciphertext, switchingKey *rlwe.SwitchingKey, ctOut *rlwe.Ciphertext)

	// Degree Management
	RelinearizeNew(ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext)
	Relinearize(ctIn *rlwe.Ciphertext, ctOut *rlwe.Ciphertext)

	// Scale Management
	ScaleUpNew(ctIn *rlwe.Ciphertext, scale rlwe.Scale) (ctOut *rlwe.Ciphertext)
	ScaleUp(ctIn *rlwe.Ciphertext, scale rlwe.Scale, ctOut *rlwe.Ciphertext)
	SetScale(ctIn *rlwe.Ciphertext, scale rlwe.Scale)
	Rescale(ctIn *rlwe.Ciphertext, minScale rlwe.Scale, ctOut *rlwe.Ciphertext) (err error)

	// Level Management
	DropLevelNew(ctIn *rlwe.Ciphertext, levels int) (ctOut *rlwe.Ciphertext)
	DropLevel(ctIn *rlwe.Ciphertext, levels int)

	// ==============
	// === Others ===
	// ==============
	GetRLWEEvaluator() *rlwe.Evaluator
	BuffQ() [3]*ring.Poly
	BuffCt() *rlwe.Ciphertext
	ShallowCopy() Evaluator
	WithKey(rlwe.EvaluationKey) Evaluator
}

// evaluator is a struct that holds the necessary elements to execute the homomorphic operations between Ciphertexts and/or Plaintexts.
// It also holds a memory buffer used to store intermediate computations.
type evaluator struct {
	*evaluatorBase
	*evaluatorBuffers
	*rlwe.Evaluator
}

type evaluatorBase struct {
	params Parameters
}

type evaluatorBuffers struct {
	buffQ  [3]*ring.Poly    // Memory buffer in order: for MForm(c0), MForm(c1), c2
	buffCt *rlwe.Ciphertext // Memory buffer for ciphertexts that need to be scaled up (to be eventually removed)
}

type collapsedlevelsInfo struct {
	collapse_depth int
	nexcollapse    [][]map[int]map[int]ringqp.Poly
	nexcollapse_pt [][]map[int]map[int][]int
	totalDiffRots  map[int]int
	level          int
	combinedmap_pt map[int][]int
	combinedmap    map[int]ringqp.Poly
}

type collapsedlevelsInfoB struct {
	collapse_depth int
	nexcollapse    [][]map[int]map[int]*ring.Poly
	nexcollapse_pt [][]map[int]map[int][]int
	totalDiffRots  map[int]int
	level          int
	scale          rlwe.Scale
	// combinedmap_pt map[int][]int
	// combinedmap    map[int]ringqp.Poly
}

// BuffQ returns a pointer to the internal memory buffer buffQ.
func (eval *evaluator) BuffQ() [3]*ring.Poly {
	return eval.buffQ
}

// BuffCt returns a pointer to the internal memory buffer buffCt.
func (eval *evaluator) BuffCt() *rlwe.Ciphertext {
	return eval.buffCt
}

func newEvaluatorBase(params Parameters) *evaluatorBase {
	ev := new(evaluatorBase)
	ev.params = params
	return ev
}

func newEvaluatorBuffers(evalBase *evaluatorBase) *evaluatorBuffers {
	buff := new(evaluatorBuffers)
	params := evalBase.params
	ringQ := params.RingQ()
	buff.buffQ = [3]*ring.Poly{ringQ.NewPoly(), ringQ.NewPoly(), ringQ.NewPoly()}
	buff.buffCt = NewCiphertext(params, 2, params.MaxLevel())
	return buff
}

// NewEvaluator creates a new Evaluator, that can be used to do homomorphic
// operations on the Ciphertexts and/or Plaintexts. It stores a memory buffer
// and Ciphertexts that will be used for intermediate values.
func NewEvaluator(params Parameters, evaluationKey rlwe.EvaluationKey) Evaluator {
	eval := new(evaluator)
	eval.evaluatorBase = newEvaluatorBase(params)
	eval.evaluatorBuffers = newEvaluatorBuffers(eval.evaluatorBase)
	eval.Evaluator = rlwe.NewEvaluator(params.Parameters, &evaluationKey)

	return eval
}

// GetRLWEEvaluator returns the underlying *rlwe.Evaluator.
func (eval *evaluator) GetRLWEEvaluator() *rlwe.Evaluator {
	return eval.Evaluator
}

func (eval *evaluator) PermuteNTTIndexesForKey(rtks *rlwe.RotationKeySet) *map[uint64][]uint64 {
	if rtks == nil {
		return &map[uint64][]uint64{}
	}
	PermuteNTTIndex := make(map[uint64][]uint64, len(rtks.Keys))
	for galEl := range rtks.Keys {
		PermuteNTTIndex[galEl] = eval.params.RingQ().PermuteNTTIndex(galEl)
	}
	return &PermuteNTTIndex
}

func (eval *evaluator) checkBinary(op0, op1, opOut rlwe.Operand, opOutMinDegree int) {
	if op0 == nil || op1 == nil || opOut == nil {
		panic("cannot checkBinary: rlwe.Operands cannot be nil")
	}

	if op0.Degree()+op1.Degree() == 0 {
		panic("cannot checkBinary: rlwe.Operands cannot be both plaintext")
	}

	if opOut.Degree() < opOutMinDegree {
		panic("cannot checkBinary: receiver rlwe.Operand degree is too small")
	}

	if !op0.El().IsNTT {
		panic("cannot checkBinary: op0 must be in NTT")
	}

	if !op1.El().IsNTT {
		panic("cannot checkBinary: op1 must be in NTT")
	}
}

func (eval *evaluator) newCiphertextBinary(op0, op1 rlwe.Operand) (ctOut *rlwe.Ciphertext) {

	maxDegree := utils.MaxInt(op0.Degree(), op1.Degree())
	minLevel := utils.MinInt(op0.Level(), op1.Level())

	return NewCiphertext(eval.params, maxDegree, minLevel)
}

// Add adds op1 to ctIn and returns the result in ctOut.
func (eval *evaluator) Add(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext) {
	eval.checkBinary(ctIn, op1, ctOut, utils.MaxInt(ctIn.Degree(), op1.Degree()))
	eval.evaluateInPlace(ctIn, op1, ctOut, eval.params.RingQ().AddLvl)
}

// AddNew adds op1 to ctIn and returns the result in a newly created element.
func (eval *evaluator) AddNew(ctIn *rlwe.Ciphertext, op1 rlwe.Operand) (ctOut *rlwe.Ciphertext) {
	ctOut = eval.newCiphertextBinary(ctIn, op1)
	eval.Add(ctIn, op1, ctOut)
	return
}

// Sub subtracts op1 from ctIn and returns the result in ctOut.
func (eval *evaluator) Sub(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext) {

	eval.checkBinary(ctIn, op1, ctOut, utils.MaxInt(ctIn.Degree(), op1.Degree()))

	eval.evaluateInPlace(ctIn, op1, ctOut, eval.params.RingQ().SubLvl)

	level := utils.MinInt(utils.MinInt(ctIn.Level(), op1.Level()), ctOut.Level())

	if ctIn.Degree() < op1.Degree() {
		for i := ctIn.Degree() + 1; i < op1.Degree()+1; i++ {
			eval.params.RingQ().NegLvl(level, ctOut.Value[i], ctOut.Value[i])
		}
	}

}

// SubNew subtracts op1 from ctIn and returns the result in a newly created element.
func (eval *evaluator) SubNew(ctIn *rlwe.Ciphertext, op1 rlwe.Operand) (ctOut *rlwe.Ciphertext) {
	ctOut = eval.newCiphertextBinary(ctIn, op1)
	eval.Sub(ctIn, op1, ctOut)
	return
}

func (eval *evaluator) evaluateInPlace(c0 *rlwe.Ciphertext, c1 rlwe.Operand, ctOut *rlwe.Ciphertext, evaluate func(int, *ring.Poly, *ring.Poly, *ring.Poly)) {

	var tmp0, tmp1 *rlwe.Ciphertext

	level := utils.MinInt(utils.MinInt(c0.Level(), c1.Level()), ctOut.Level())

	maxDegree := utils.MaxInt(c0.Degree(), c1.Degree())
	minDegree := utils.MinInt(c0.Degree(), c1.Degree())

	// Else resizes the receiver element
	ctOut.El().Resize(maxDegree, ctOut.Level())

	c0Scale := c0.GetScale().Float64()
	c1Scale := c1.GetScale().Float64()

	if ctOut.Level() > level {
		eval.DropLevel(ctOut, ctOut.Level()-utils.MinInt(c0.Level(), c1.Level()))
	}

	cmp := c0.GetScale().Cmp(c1.GetScale())

	// Checks whether or not the receiver element is the same as one of the input elements
	// and acts accordingly to avoid unnecessary element creation or element overwriting,
	// and scales properly the element before the evaluation.
	if ctOut == c0 {

		if cmp == 1 && math.Floor(c0Scale/c1Scale) > 1 {

			tmp1 = eval.buffCt.El()
			tmp1.Scale = ctOut.Scale

			eval.MultByConst(c1.El(), math.Floor(c0Scale/c1Scale), tmp1)

		} else if cmp == -1 && math.Floor(c1Scale/c0Scale) > 1 {

			eval.MultByConst(c0, math.Floor(c1Scale/c0Scale), c0)

			ctOut.Scale = c1.GetScale()

			tmp1 = c1.El()
		} else {
			tmp1 = c1.El()
		}

		tmp0 = c0.El()

	} else if ctOut == c1 {

		if cmp == 1 && math.Floor(c0Scale/c1Scale) > 1 {

			eval.MultByConst(c1.El(), math.Floor(c0Scale/c1Scale), ctOut)

			ctOut.Scale = c0.Scale

			tmp0 = c0.El()

		} else if cmp == -1 && math.Floor(c1Scale/c0Scale) > 1 {

			tmp0 = eval.buffCt.El()
			tmp0.Scale = ctOut.Scale

			eval.MultByConst(c0, math.Floor(c1Scale/c0Scale), tmp0)
		} else {
			tmp0 = c0.El()
		}

		tmp1 = c1.El()

	} else {

		if cmp == 1 && math.Floor(c0Scale/c1Scale) > 1 {

			tmp1 = eval.buffCt.El()

			tmp1.Scale = ctOut.Scale

			eval.MultByConst(c1.El(), math.Floor(c0Scale/c1Scale), tmp1)

			tmp0 = c0.El()

		} else if cmp == -1 && math.Floor(c1Scale/c0Scale) > 1 {

			tmp0 = eval.buffCt.El()

			tmp0.Scale = ctOut.Scale

			eval.MultByConst(c0, math.Floor(c1Scale/c0Scale), tmp0)

			tmp1 = c1.El()

		} else {
			tmp0 = c0.El()
			tmp1 = c1.El()
		}
	}

	for i := 0; i < minDegree+1; i++ {
		evaluate(level, tmp0.Value[i], tmp1.Value[i], ctOut.El().Value[i])
	}

	ctOut.MetaData = c0.MetaData
	ctOut.Scale = c0.Scale.Max(c1.GetScale())

	// If the inputs degrees differ, it copies the remaining degree on the receiver.
	// Also checks that the receiver is not one of the inputs to avoid unnecessary work.

	if c0.Degree() > c1.Degree() && tmp0 != ctOut.El() {
		for i := minDegree + 1; i < maxDegree+1; i++ {
			ring.CopyLvl(level, tmp0.Value[i], ctOut.El().Value[i])
		}
	} else if c1.Degree() > c0.Degree() && tmp1 != ctOut.El() {
		for i := minDegree + 1; i < maxDegree+1; i++ {
			ring.CopyLvl(level, tmp1.Value[i], ctOut.El().Value[i])
		}
	}
}

// Neg negates the value of ct0 and returns the result in ctOut.
func (eval *evaluator) Neg(ct0 *rlwe.Ciphertext, ctOut *rlwe.Ciphertext) {

	level := utils.MinInt(ct0.Level(), ctOut.Level())

	if ct0.Degree() != ctOut.Degree() {
		panic("cannot Negate: invalid receiver Ciphertext does not match input Ciphertext degree")
	}

	for i := range ct0.Value {
		eval.params.RingQ().NegLvl(level, ct0.Value[i], ctOut.Value[i])
	}

	ctOut.MetaData = ct0.MetaData
}

// NegNew negates ct0 and returns the result in a newly created element.
func (eval *evaluator) NegNew(ct0 *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext) {
	ctOut = NewCiphertext(eval.params, ct0.Degree(), ct0.Level())
	eval.Neg(ct0, ctOut)
	return
}

// AddConstNew adds the input constant (which can be a uint64, int64, float64 or complex128) to ct0 and returns the result in a new element.
func (eval *evaluator) AddConstNew(ct0 *rlwe.Ciphertext, constant interface{}) (ctOut *rlwe.Ciphertext) {
	ctOut = ct0.CopyNew()
	eval.AddConst(ct0, constant, ctOut)
	return ctOut
}

func (eval *evaluator) getConstAndScale(level int, constant interface{}) (cReal, cImag, scale float64) {

	// Converts to float64 and determines if a scaling is required (which is the case if either real or imag have a rational part)
	scale = 1
	switch constant := constant.(type) {
	case complex128:
		cReal = real(constant)
		cImag = imag(constant)

		if cReal != 0 {
			valueInt := int64(cReal)
			valueFloat := cReal - float64(valueInt)

			if valueFloat != 0 {
				scale = float64(eval.params.RingQ().Modulus[level])
			}
		}

		if cImag != 0 {
			valueInt := int64(cImag)
			valueFloat := cImag - float64(valueInt)

			if valueFloat != 0 {
				scale = float64(eval.params.RingQ().Modulus[level])
			}
		}

	case float64:
		cReal = constant
		cImag = float64(0)

		if cReal != 0 {
			valueInt := int64(cReal)
			valueFloat := cReal - float64(valueInt)

			if valueFloat != 0 {
				scale = float64(eval.params.RingQ().Modulus[level])
			}
		}

	case *big.Float:
		cf64, _ := constant.Float64()
		return eval.getConstAndScale(level, cf64)

	case uint64:
		cReal = float64(constant)
		cImag = float64(0)

	case int64:
		cReal = float64(constant)
		cImag = float64(0)

	case int:
		cReal = float64(constant)
		cImag = float64(0)
	}

	if eval.params.RingType() == ring.ConjugateInvariant {
		cImag = float64(0)
	}

	return
}

// AddConst adds the input constant (which can be a uint64, int64, float64 or complex128) to ct0 and returns the result in ctOut.
func (eval *evaluator) AddConst(ct0 *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext) {

	var level = utils.MinInt(ct0.Level(), ctOut.Level())
	var scaledConst, scaledConstReal, scaledConstImag, qi uint64

	cReal, cImag, _ := eval.getConstAndScale(level, constant)

	ringQ := eval.params.RingQ()

	cf64 := ctOut.Scale.Float64()

	ctOut.MetaData = ct0.MetaData

	// Component wise addition of the following vector to the ciphertext:
	// [a + b*psi_qi^2, ....., a + b*psi_qi^2, a - b*psi_qi^2, ...., a - b*psi_qi^2] mod Qi
	// [{                  N/2                }{                N/2               }]
	// Which is equivalent outside of the NTT domain to adding a to the first coefficient of ct0 and b to the N/2-th coefficient of ct0.
	for i := 0; i < level+1; i++ {
		scaledConstReal, scaledConstImag, scaledConst = 0, 0, 0
		qi = ringQ.Modulus[i]

		if cReal != 0 {
			scaledConstReal = scaleUpExact(cReal, cf64, qi)
			scaledConst = scaledConstReal
		}

		if cImag != 0 {
			scaledConstImag = ring.MRed(scaleUpExact(cImag, cf64, qi), ringQ.NttPsi[i][1], qi, ringQ.MredParams[i])
			scaledConst = ring.CRed(scaledConst+scaledConstImag, qi)
		}

		p0tmp := ct0.Value[0].Coeffs[i]
		p1tmp := ctOut.Value[0].Coeffs[i]

		ring.AddScalarVec(p0tmp[:ringQ.N>>1], p1tmp[:ringQ.N>>1], scaledConst, qi)

		if cImag != 0 {
			scaledConst = ring.CRed(scaledConstReal+(qi-scaledConstImag), qi)
		}

		ring.AddScalarVec(p0tmp[ringQ.N>>1:], p1tmp[ringQ.N>>1:], scaledConst, qi)
	}
}

// MultByConstAndAdd multiplies ct0 by the input constant, and adds it to the receiver element (it does not modify the input
// element), e.g., ctOut(x) = ctOut(x) + ct0(x) * (a+bi). This functions removes the need of storing the intermediate value c(x) * (a+bi).
// This function will modify the level and the scale of the receiver element depending on the level and the scale of the input
// element and the type of the constant. The level of the receiver element will be set to min(input.level, receiver.level).
// The scale of the receiver element will be set to the scale that the input element would have after the multiplication by the constant.
func (eval *evaluator) MultByConstAndAdd(ct0 *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext) {

	var level = utils.MinInt(ct0.Level(), ctOut.Level())

	// Forces a drop of ctOut level to ct0 level
	if ctOut.Level() > level {
		eval.DropLevel(ctOut, ctOut.Level()-level)
	}

	cReal, cImag, scale := eval.getConstAndScale(level, constant)

	var scaledConst, scaledConstReal, scaledConstImag uint64

	c0f64 := ct0.Scale.Float64()
	c1f64 := ctOut.Scale.Float64()

	ringQ := eval.params.RingQ()

	// If a scaling would be required to multiply by the constant,
	// it equalizes scales such that the scales match in the end.
	if scale != 1 {

		// If ctOut scaling is smaller than ct0's scale + the default scaling,
		// then brings ctOut scale to ct0's scale.
		if c1f64 < c0f64*scale {

			if scale := math.Floor((scale * c0f64) / c1f64); scale > 1 {

				eval.MultByConst(ctOut, scale, ctOut)

			}

			ctOut.MetaData = ct0.MetaData
			ctOut.Scale = ct0.Scale.Mul(rlwe.NewScale(scale))

			// If ctOut.scale > ((a+bi)*scale)*ct0(x), then it sets the scale to
			// bring c(x)*scale to the level of ctOut(x) scale
		} else if c1f64 > c0f64*scale {
			scale = c1f64 / c0f64
		}

		// If no scaling is required, then it sets the appropriate scale such that
		// ct0(x)*scale matches ctOut(x) scale without modifying ct0(x) scale.
	} else {

		if c1f64 > c0f64 {

			scale = c1f64 / c0f64

		} else if c0f64 > c1f64 {

			if scale := math.Floor(c0f64 / c1f64); scale > 1 {
				eval.MultByConst(ctOut, scale, ctOut)
			}

			ctOut.MetaData = ct0.MetaData
			ctOut.Scale = ct0.Scale
		}
	}

	// Component-wise multiplication of the following vector to the ciphertext:
	// [a + b*psi_qi^2, ....., a + b*psi_qi^2, a - b*psi_qi^2, ...., a - b*psi_qi^2] mod Qi
	// [{                  N/2                }{                N/2               }]
	// Which is equivalent outside of the NTT domain to adding a to the first coefficient of ct0 and b to the N/2-th coefficient of ct0.
	for i := 0; i < level+1; i++ {

		qi := ringQ.Modulus[i]
		mredParams := ringQ.MredParams[i]
		bredParams := ringQ.BredParams[i]

		scaledConstReal = 0
		scaledConstImag = 0
		scaledConst = 0

		if cReal != 0 {
			scaledConstReal = scaleUpExact(cReal, scale, qi)
			scaledConst = scaledConstReal
		}

		if cImag != 0 {
			scaledConstImag = scaleUpExact(cImag, scale, qi)
			scaledConstImag = ring.MRed(scaledConstImag, ringQ.NttPsi[i][1], qi, mredParams)
			scaledConst = ring.CRed(scaledConst+scaledConstImag, qi)
		}

		scaledConst = ring.MForm(scaledConst, qi, bredParams)

		for u := range ct0.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryAndAddVec(p0tmp[:ringQ.N>>1], p1tmp[:ringQ.N>>1], scaledConst, qi, mredParams)
		}

		if cImag != 0 {
			scaledConst = ring.CRed(scaledConstReal+(qi-scaledConstImag), qi)
			scaledConst = ring.MForm(scaledConst, qi, bredParams)
		}

		for u := range ct0.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryAndAddVec(p0tmp[ringQ.N>>1:], p1tmp[ringQ.N>>1:], scaledConst, qi, mredParams)
		}
	}
}

// MultByConstNew multiplies ct0 by the input constant and returns the result in a newly created element.
// The scale of the output element will depend on the scale of the input element and the constant (if the constant
// needs to be scaled (its rational part is not zero)). The constant can be a uint64, int64, float64 or complex128.
func (eval *evaluator) MultByConstNew(ct0 *rlwe.Ciphertext, constant interface{}) (ctOut *rlwe.Ciphertext) {
	ctOut = NewCiphertext(eval.params, ct0.Degree(), ct0.Level())
	eval.MultByConst(ct0, constant, ctOut)
	return
}

// MultByConst multiplies ct0 by the input constant and returns the result in ctOut.
// The scale of the output element will depend on the scale of the input element and the constant (if the constant
// needs to be scaled (its rational part is not zero)). The constant can be a uint64, int64, float64 or complex128.
func (eval *evaluator) MultByConst(ct0 *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext) {

	var level = utils.MinInt(ct0.Level(), ctOut.Level())

	cReal, cImag, scale := eval.getConstAndScale(level, constant)

	// Component wise multiplication of the following vector with the ciphertext:
	// [a + b*psi_qi^2, ....., a + b*psi_qi^2, a - b*psi_qi^2, ...., a - b*psi_qi^2] mod Qi
	// [{                  N/2                }{                N/2               }]
	// Which is equivalent outside of the NTT domain to adding a to the first coefficient of ct0 and b to the N/2-th coefficient of ct0.
	ringQ := eval.params.RingQ()
	var scaledConst, scaledConstReal, scaledConstImag uint64
	for i := 0; i < level+1; i++ {

		qi := ringQ.Modulus[i]
		bredParams := ringQ.BredParams[i]
		mredParams := ringQ.MredParams[i]

		scaledConstReal = 0
		scaledConstImag = 0
		scaledConst = 0

		if cReal != 0 {
			scaledConstReal = scaleUpExact(cReal, scale, qi)
			scaledConst = scaledConstReal
		}

		if cImag != 0 {
			scaledConstImag = scaleUpExact(cImag, scale, qi)
			scaledConstImag = ring.MRed(scaledConstImag, ringQ.NttPsi[i][1], qi, mredParams)
			scaledConst = ring.CRed(scaledConst+scaledConstImag, qi)
		}

		scaledConst = ring.MForm(scaledConst, qi, bredParams)

		for u := range ct0.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryVec(p0tmp[:ringQ.N>>1], p1tmp[:ringQ.N>>1], scaledConst, qi, mredParams)
		}

		if cImag != 0 {
			scaledConst = ring.CRed(scaledConstReal+(qi-scaledConstImag), qi)
			scaledConst = ring.MForm(scaledConst, qi, bredParams)
		}

		for u := range ct0.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryVec(p0tmp[ringQ.N>>1:], p1tmp[ringQ.N>>1:], scaledConst, qi, mredParams)
		}
	}

	ctOut.MetaData = ct0.MetaData
	ctOut.Scale = ct0.Scale.Mul(rlwe.NewScale(scale))
}

// MultByGaussianInteger multiples the ct0 by the gaussian integer cReal + i*cImag and returns the result on ctOut.
// Accepted types for cReal and cImag are uint64, int64 and big.Int.
func (eval *evaluator) MultByGaussianInteger(ct0 *rlwe.Ciphertext, cReal, cImag interface{}, ctOut *rlwe.Ciphertext) {

	ringQ := eval.params.RingQ()

	level := utils.MinInt(ct0.Level(), ctOut.Level())
	var scaledConst, scaledConstReal, scaledConstImag uint64

	ctOut.MetaData = ct0.MetaData

	for i := 0; i < level+1; i++ {

		qi := ringQ.Modulus[i]
		bredParams := ringQ.BredParams[i]
		mredParams := ringQ.MredParams[i]

		scaledConstReal = interfaceMod(cReal, qi)

		if eval.params.RingType() != ring.ConjugateInvariant {
			scaledConstImag = interfaceMod(cImag, qi)
		}

		scaledConst = scaledConstReal

		if scaledConstImag != 0 {
			scaledConstImag = ring.MRed(scaledConstImag, ringQ.NttPsi[i][1], qi, mredParams)
			scaledConst = ring.CRed(scaledConst+scaledConstImag, qi)
		}

		scaledConst = ring.MForm(scaledConst, qi, bredParams)

		for u := range ct0.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryVec(p0tmp[:ringQ.N>>1], p1tmp[:ringQ.N>>1], scaledConst, qi, mredParams)
		}

		if cImag != 0 {
			scaledConst = ring.CRed(scaledConstReal+(qi-scaledConstImag), qi)
			scaledConst = ring.MForm(scaledConst, qi, bredParams)
		}

		for u := range ct0.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryVec(p0tmp[ringQ.N>>1:], p1tmp[ringQ.N>>1:], scaledConst, qi, mredParams)
		}
	}
}

// MultByGaussianIntegerAndAdd multiples the ct0 by the gaussian integer cReal + i*cImag and adds the result on ctOut.
// Accepted types for cReal and cImag are uint64, int64 and big.Int.
func (eval *evaluator) MultByGaussianIntegerAndAdd(ct0 *rlwe.Ciphertext, cReal, cImag interface{}, ctOut *rlwe.Ciphertext) {

	ringQ := eval.params.RingQ()

	level := utils.MinInt(ct0.Level(), ctOut.Level())
	var scaledConst, scaledConstReal, scaledConstImag uint64

	for i := 0; i < level+1; i++ {

		qi := ringQ.Modulus[i]
		bredParams := ringQ.BredParams[i]
		mredParams := ringQ.MredParams[i]

		scaledConstReal = interfaceMod(cReal, qi)

		if eval.params.RingType() != ring.ConjugateInvariant {
			scaledConstImag = interfaceMod(cImag, qi)
		}

		scaledConst = scaledConstReal

		if scaledConstImag != 0 {
			scaledConstImag = ring.MRed(scaledConstImag, ringQ.NttPsi[i][1], qi, mredParams)
			scaledConst = ring.CRed(scaledConst+scaledConstImag, qi)
		}

		scaledConst = ring.MForm(scaledConst, qi, bredParams)

		for u := range ct0.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryAndAddVec(p0tmp[:ringQ.N>>1], p1tmp[:ringQ.N>>1], scaledConst, qi, mredParams)
		}

		if cImag != 0 {
			scaledConst = ring.CRed(scaledConstReal+(qi-scaledConstImag), qi)
			scaledConst = ring.MForm(scaledConst, qi, bredParams)
		}

		for u := range ct0.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryAndAddVec(p0tmp[ringQ.N>>1:], p1tmp[ringQ.N>>1:], scaledConst, qi, mredParams)
		}
	}
}

// MultByiNew multiplies ct0 by the imaginary number i, and returns the result in a newly created element.
// It does not change the scale.
func (eval *evaluator) MultByiNew(ct0 *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext) {

	if eval.params.RingType() == ring.ConjugateInvariant {
		panic("cannot MultByiNew: method not supported when params.RingType() == ring.ConjugateInvariant")
	}

	ctOut = NewCiphertext(eval.params, 1, ct0.Level())
	eval.MultByi(ct0, ctOut)
	return ctOut
}

// MultByi multiplies ct0 by the imaginary number i, and returns the result in ctOut.
// It does not change the scale.
func (eval *evaluator) MultByi(ct0 *rlwe.Ciphertext, ctOut *rlwe.Ciphertext) {

	if eval.params.RingType() == ring.ConjugateInvariant {
		panic("cannot MultByi: method not supported when params.RingType() == ring.ConjugateInvariant")
	}

	var level = utils.MinInt(ct0.Level(), ctOut.Level())
	ctOut.MetaData = ct0.MetaData

	ringQ := eval.params.RingQ()

	var imag uint64

	// Equivalent to a product by the monomial x^(n/2) outside of the NTT domain
	for i := 0; i < level+1; i++ {

		qi := ringQ.Modulus[i]
		mredParams := ringQ.MredParams[i]

		imag = ringQ.NttPsi[i][1] // Psi^2

		for u := range ctOut.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryVec(p0tmp[:ringQ.N>>1], p1tmp[:ringQ.N>>1], imag, qi, mredParams)
		}

		imag = qi - imag

		for u := range ctOut.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryVec(p0tmp[ringQ.N>>1:], p1tmp[ringQ.N>>1:], imag, qi, mredParams)
		}
	}
}

// DivByiNew multiplies ct0 by the imaginary number 1/i = -i, and returns the result in a newly created element.
// It does not change the scale.
func (eval *evaluator) DivByiNew(ct0 *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext) {

	if eval.params.RingType() == ring.ConjugateInvariant {
		panic("cannot DivByiNew: method not supported when params.RingType() == ring.ConjugateInvariant")
	}

	ctOut = NewCiphertext(eval.params, 1, ct0.Level())
	eval.DivByi(ct0, ctOut)
	return
}

// DivByi multiplies ct0 by the imaginary number 1/i = -i, and returns the result in ctOut.
// It does not change the scale.
func (eval *evaluator) DivByi(ct0 *rlwe.Ciphertext, ctOut *rlwe.Ciphertext) {

	if eval.params.RingType() == ring.ConjugateInvariant {
		panic("cannot DivByi: method not supported when params.RingType() == ring.ConjugateInvariant")
	}

	var level = utils.MinInt(ct0.Level(), ctOut.Level())

	ringQ := eval.params.RingQ()

	ctOut.MetaData = ct0.MetaData

	var imag uint64

	// Equivalent to a product by the monomial x^(3*n/2) outside of the NTT domain
	for i := 0; i < level+1; i++ {

		qi := ringQ.Modulus[i]
		mredParams := ringQ.MredParams[i]

		imag = qi - ringQ.NttPsi[i][1] // -Psi^2

		for u := range ctOut.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryVec(p0tmp[:ringQ.N>>1], p1tmp[:ringQ.N>>1], imag, qi, mredParams)
		}

		imag = ringQ.NttPsi[i][1] // Psi^2

		for u := range ctOut.Value {
			p0tmp := ct0.Value[u].Coeffs[i]
			p1tmp := ctOut.Value[u].Coeffs[i]
			ring.MulScalarMontgomeryVec(p0tmp[ringQ.N>>1:], p1tmp[ringQ.N>>1:], imag, qi, mredParams)
		}
	}
}

// ScaleUpNew multiplies ct0 by 2^scale and sets its scale to its previous scale
// plus 2^n. It returns the result in a newly created element.
func (eval *evaluator) ScaleUpNew(ct0 *rlwe.Ciphertext, scale rlwe.Scale) (ctOut *rlwe.Ciphertext) {
	ctOut = NewCiphertext(eval.params, ct0.Degree(), ct0.Level())
	eval.ScaleUp(ct0, scale, ctOut)
	return
}

// ScaleUp multiplies ct0 by 2^scale and sets its scale to its previous scale
// plus 2^n. It returns the result in ctOut.
func (eval *evaluator) ScaleUp(ct0 *rlwe.Ciphertext, scale rlwe.Scale, ctOut *rlwe.Ciphertext) {
	eval.MultByConst(ct0, scale.Uint64(), ctOut)
	ctOut.MetaData = ct0.MetaData
	ctOut.Scale = ct0.Scale.Mul(scale)
}

// SetScale sets the scale of the ciphertext to the input scale (consumes a level)
func (eval *evaluator) SetScale(ct *rlwe.Ciphertext, scale rlwe.Scale) {

	eval.MultByConst(ct, scale.Float64()/ct.Scale.Float64(), ct)
	if err := eval.Rescale(ct, scale, ct); err != nil {
		panic(err)
	}
	ct.Scale = scale
}

// DropLevelNew reduces the level of ct0 by levels and returns the result in a newly created element.
// No rescaling is applied during this procedure.
func (eval *evaluator) DropLevelNew(ct0 *rlwe.Ciphertext, levels int) (ctOut *rlwe.Ciphertext) {
	ctOut = ct0.CopyNew()
	eval.DropLevel(ctOut, levels)
	return
}

// DropLevel reduces the level of ct0 by levels and returns the result in ct0.
// No rescaling is applied during this procedure.
func (eval *evaluator) DropLevel(ct0 *rlwe.Ciphertext, levels int) {
	ct0.Resize(ct0.Degree(), ct0.Level()-levels)
}

// RescaleNew divides ct0 by the last modulus in the moduli chain, and repeats this
// procedure (consuming one level each time) until the scale reaches the original scale or before it goes below it, and returns the result
// in a newly created element. Since all the moduli in the moduli chain are generated to be close to the
// original scale, this procedure is equivalent to dividing the input element by the scale and adding
// some error.
// Returns an error if "threshold <= 0", ct.scale = 0, ct.Level() = 0, ct.IsNTT() != true
func (eval *evaluator) RescaleNew(ct0 *rlwe.Ciphertext, minScale rlwe.Scale) (ctOut *rlwe.Ciphertext, err error) {

	ctOut = NewCiphertext(eval.params, ct0.Degree(), ct0.Level())

	return ctOut, eval.Rescale(ct0, minScale, ctOut)
}

// Rescale divides ct0 by the last modulus in the moduli chain, and repeats this
// procedure (consuming one level each time) until the scale reaches the original scale or before it goes below it, and returns the result
// in ctOut. Since all the moduli in the moduli chain are generated to be close to the
// original scale, this procedure is equivalent to dividing the input element by the scale and adding
// some error.
// Returns an error if "minScale <= 0", ct.scale = 0, ct.Level() = 0, ct.IsNTT() != true or if ct.Leve() != ctOut.Level()
func (eval *evaluator) Rescale(ctIn *rlwe.Ciphertext, minScale rlwe.Scale, ctOut *rlwe.Ciphertext) (err error) {

	ringQ := eval.params.RingQ()

	if minScale.Cmp(rlwe.NewScale(0)) != 1 {
		return errors.New("cannot Rescale: minScale is <0")
	}

	minScale = minScale.Div(rlwe.NewScale(2))

	if ctIn.Scale.Cmp(rlwe.NewScale(0)) != 1 {
		return errors.New("cannot Rescale: ciphertext scale is <0")
	}

	if ctIn.Level() == 0 {
		return errors.New("cannot Rescale: input Ciphertext already at level 0")
	}

	if ctOut.Degree() != ctIn.Degree() {
		return errors.New("cannot Rescale: ctIn.Degree() != ctOut.Degree()")
	}

	ctOut.MetaData = ctIn.MetaData

	currentLevel := ctIn.Level()

	// Divides the scale by each moduli of the modulus chain as long as the scale isn't smaller than minScale/2
	// or until the output Level() would be zero
	var nbRescales int
	for currentLevel >= 0 {

		scale := ctOut.Scale.Div(rlwe.NewScale(ringQ.Modulus[currentLevel]))

		if scale.Cmp(minScale) == -1 {
			break
		}

		ctOut.Scale = scale

		nbRescales++
		currentLevel--
	}

	if nbRescales > 0 {
		level := ctIn.Level()
		for i := range ctOut.Value {
			ringQ.DivRoundByLastModulusManyNTTLvl(level, nbRescales, ctIn.Value[i], eval.buffQ[0], ctOut.Value[i])
		}
		ctOut.Resize(ctOut.Degree(), level-nbRescales)
	} else {
		if ctIn != ctOut {
			ctOut.Copy(ctIn)
		}
	}

	return nil
}

// MulNew multiplies ctIn with op1 without relinearization and returns the result in a newly created element.
// The procedure will panic if either ctIn.Degree or op1.Degree > 1.
func (eval *evaluator) MulNew(ctIn *rlwe.Ciphertext, op1 rlwe.Operand) (ctOut *rlwe.Ciphertext) {
	ctOut = NewCiphertext(eval.params, ctIn.Degree()+op1.Degree(), utils.MinInt(ctIn.Level(), op1.Level()))
	eval.mulRelin(ctIn, op1, false, ctOut)
	return
}

// Mul multiplies ctIn with op1 without relinearization and returns the result in ctOut.
// The procedure will panic if either ctIn or op1 are have a degree higher than 1.
// The procedure will panic if ctOut.Degree != ctIn.Degree + op1.Degree.
func (eval *evaluator) Mul(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext) {
	eval.mulRelin(ctIn, op1, false, ctOut)
}

// MulRelinNew multiplies ctIn with op1 with relinearization and returns the result in a newly created element.
// The procedure will panic if either ctIn.Degree or op1.Degree > 1.
// The procedure will panic if the evaluator was not created with an relinearization key.
func (eval *evaluator) MulRelinNew(ctIn *rlwe.Ciphertext, op1 rlwe.Operand) (ctOut *rlwe.Ciphertext) {
	ctOut = NewCiphertext(eval.params, 1, utils.MinInt(ctIn.Level(), op1.Level()))
	eval.mulRelin(ctIn, op1, true, ctOut)
	return
}

// MulRelin multiplies ctIn with op1 with relinearization and returns the result in ctOut.
// The procedure will panic if either ctIn.Degree or op1.Degree > 1.
// The procedure will panic if ctOut.Degree != ctIn.Degree + op1.Degree.
// The procedure will panic if the evaluator was not created with an relinearization key.
func (eval *evaluator) MulRelin(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext) {
	eval.mulRelin(ctIn, op1, true, ctOut)
}

func (eval *evaluator) mulRelin(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, relin bool, ctOut *rlwe.Ciphertext) {

	level := utils.MinInt(utils.MinInt(ctIn.Level(), op1.Level()), ctOut.Level())

	if ctIn.Degree()+op1.Degree() > 2 {
		panic("cannot MulRelin: the sum of the input elements' total degree cannot be larger than 2")
	}

	ctOut.MetaData = ctIn.MetaData
	ctOut.Scale = ctIn.Scale.Mul(op1.GetScale())

	ringQ := eval.params.RingQ()

	var c00, c01, c0, c1, c2 *ring.Poly

	// Case Ciphertext (x) Ciphertext
	if ctIn.Degree() == 1 && op1.Degree() == 1 {

		c00 = eval.buffQ[0]
		c01 = eval.buffQ[1]

		c0 = ctOut.Value[0]
		c1 = ctOut.Value[1]

		if !relin {
			ctOut.El().Resize(2, level)
			c2 = ctOut.Value[2]
		} else {
			ctOut.El().Resize(1, level)
			c2 = eval.buffQ[2]
		}

		eval.checkBinary(ctIn, op1, ctOut, ctOut.Degree())

		// Avoid overwriting if the second input is the output
		var tmp0, tmp1 *rlwe.Ciphertext
		if op1.El() == ctOut.El() {
			tmp0, tmp1 = op1.El(), ctIn.El()
		} else {
			tmp0, tmp1 = ctIn.El(), op1.El()
		}

		ringQ.MFormLvl(level, tmp0.Value[0], c00)
		ringQ.MFormLvl(level, tmp0.Value[1], c01)

		if ctIn.El() == op1.El() { // squaring case
			ringQ.MulCoeffsMontgomeryLvl(level, c00, tmp1.Value[0], c0) // c0 = c[0]*c[0]
			ringQ.MulCoeffsMontgomeryLvl(level, c01, tmp1.Value[1], c2) // c2 = c[1]*c[1]
			ringQ.MulCoeffsMontgomeryLvl(level, c00, tmp1.Value[1], c1) // c1 = 2*c[0]*c[1]
			ringQ.AddLvl(level, c1, c1, c1)

		} else { // regular case
			ringQ.MulCoeffsMontgomeryLvl(level, c00, tmp1.Value[0], c0) // c0 = c0[0]*c0[0]
			ringQ.MulCoeffsMontgomeryLvl(level, c01, tmp1.Value[1], c2) // c2 = c0[1]*c1[1]
			ringQ.MulCoeffsMontgomeryLvl(level, c00, tmp1.Value[1], c1)
			ringQ.MulCoeffsMontgomeryAndAddLvl(level, c01, tmp1.Value[0], c1) // c1 = c0[0]*c1[1] + c0[1]*c1[0]
		}

		if relin {

			if eval.Rlk == nil {
				panic("cannot MulRelin: relinearization key is missing")
			}

			tmpCt := &rlwe.Ciphertext{Value: []*ring.Poly{eval.BuffQP[1].Q, eval.BuffQP[2].Q}}
			tmpCt.IsNTT = true

			eval.GadgetProduct(level, c2, eval.Rlk.Keys[0].GadgetCiphertext, tmpCt)
			ringQ.AddLvl(level, c0, tmpCt.Value[0], ctOut.Value[0])
			ringQ.AddLvl(level, c1, tmpCt.Value[1], ctOut.Value[1])
		}

		// Case Plaintext (x) Ciphertext or Ciphertext (x) Plaintext
	} else {

		eval.checkBinary(ctIn, op1, ctOut, ctOut.Degree())

		var c0 *ring.Poly
		var c1 []*ring.Poly
		if ctIn.Degree() == 0 {
			c0 = eval.buffQ[0]
			ringQ.MFormLvl(level, ctIn.Value[0], c0)
			c1 = op1.El().Value

		} else {
			c0 = eval.buffQ[0]
			ringQ.MFormLvl(level, op1.El().Value[0], c0)
			c1 = ctIn.Value
		}

		ctOut.El().Resize(ctIn.Degree()+op1.Degree(), level)

		for i := range c1 {
			ringQ.MulCoeffsMontgomeryLvl(level, c0, c1[i], ctOut.Value[i])
		}
	}
}

// MulAndAdd multiplies ctIn with op1 without relinearization and adds the result on ctOut.
// User must ensure that ctOut.scale <= ctIn.scale * op1.scale.
// If ctOut.scale < ctIn.scale * op1.scale, then scales up ctOut before adding the result.
// The procedure will panic if either ctIn or op1 are have a degree higher than 1.
// The procedure will panic if ctOut.Degree != ctIn.Degree + op1.Degree.
// The procedure will panic if ctOut = ctIn or op1.
func (eval *evaluator) MulAndAdd(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext) {
	eval.mulRelinAndAdd(ctIn, op1, false, ctOut)
}

// MulRelinAndAdd multiplies ctIn with op1 with relinearization and adds the result on ctOut.
// User must ensure that ctOut.scale <= ctIn.scale * op1.scale.
// If ctOut.scale < ctIn.scale * op1.scale, then scales up ctOut before adding the result.
// The procedure will panic if either ctIn.Degree or op1.Degree > 1.
// The procedure will panic if ctOut.Degree != ctIn.Degree + op1.Degree.
// The procedure will panic if the evaluator was not created with an relinearization key.
// The procedure will panic if ctOut = ctIn or op1.
func (eval *evaluator) MulRelinAndAdd(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, ctOut *rlwe.Ciphertext) {
	eval.mulRelinAndAdd(ctIn, op1, true, ctOut)
}

func (eval *evaluator) mulRelinAndAdd(ctIn *rlwe.Ciphertext, op1 rlwe.Operand, relin bool, ctOut *rlwe.Ciphertext) {

	eval.checkBinary(ctIn, op1, ctOut, utils.MaxInt(ctIn.Degree(), op1.Degree()))

	level := utils.MinInt(utils.MinInt(ctIn.Level(), op1.Level()), ctOut.Level())

	if ctIn.Degree()+op1.Degree() > 2 {
		panic("cannot MulRelinAndAdd: the sum of the input elements' degree cannot be larger than 2")
	}

	if ctIn.El() == ctOut.El() || op1.El() == ctOut.El() {
		panic("cannot MulRelinAndAdd: ctOut must be different from op0 and op1")
	}

	c0f64 := ctIn.Scale.Float64()
	c1f64 := op1.GetScale().Float64()
	c2f64 := ctOut.Scale.Float64()

	resScale := c0f64 * c1f64

	if c2f64 < resScale {
		eval.MultByConst(ctOut, math.Round(resScale/c2f64), ctOut)
		ctOut.Scale = rlwe.NewScale(resScale)
	}

	ringQ := eval.params.RingQ()

	var c00, c01, c0, c1, c2 *ring.Poly

	// Case Ciphertext (x) Ciphertext
	if ctIn.Degree() == 1 && op1.Degree() == 1 {

		c00 = eval.buffQ[0]
		c01 = eval.buffQ[1]

		c0 = ctOut.Value[0]
		c1 = ctOut.Value[1]

		if !relin {
			ctOut.El().Resize(2, level)
			c2 = ctOut.Value[2]
		} else {
			// No resize here since we add on ctOut
			c2 = eval.buffQ[2]
		}

		tmp0, tmp1 := ctIn.El(), op1.El()

		ringQ.MFormLvl(level, tmp0.Value[0], c00)
		ringQ.MFormLvl(level, tmp0.Value[1], c01)

		ringQ.MulCoeffsMontgomeryAndAddLvl(level, c00, tmp1.Value[0], c0) // c0 += c[0]*c[0]
		ringQ.MulCoeffsMontgomeryAndAddLvl(level, c00, tmp1.Value[1], c1) // c1 += c[0]*c[1]
		ringQ.MulCoeffsMontgomeryAndAddLvl(level, c01, tmp1.Value[0], c1) // c1 += c[1]*c[0]

		if relin {

			if eval.Rlk == nil {
				panic("cannot MulRelinAndAdd: relinearization key is missing")
			}

			ringQ.MulCoeffsMontgomeryLvl(level, c01, tmp1.Value[1], c2) // c2 += c[1]*c[1]

			tmpCt := &rlwe.Ciphertext{Value: []*ring.Poly{eval.BuffQP[1].Q, eval.BuffQP[2].Q}}
			tmpCt.IsNTT = true

			eval.GadgetProduct(level, c2, eval.Rlk.Keys[0].GadgetCiphertext, tmpCt)
			ringQ.AddLvl(level, c0, tmpCt.Value[0], c0)
			ringQ.AddLvl(level, c1, tmpCt.Value[1], c1)
		} else {
			ringQ.MulCoeffsMontgomeryAndAddLvl(level, c01, tmp1.Value[1], c2) // c2 += c[1]*c[1]
		}

		// Case Plaintext (x) Ciphertext or Ciphertext (x) Plaintext
	} else {

		if ctOut.Degree() < ctIn.Degree() {
			ctOut.Resize(ctIn.Degree(), level)
		}

		c00 := eval.buffQ[0]

		ringQ.MFormLvl(level, op1.El().Value[0], c00)
		for i := range ctIn.Value {
			ringQ.MulCoeffsMontgomeryAndAddLvl(level, ctIn.Value[i], c00, ctOut.Value[i])
		}
	}
}

// RelinearizeNew applies the relinearization procedure on ct0 and returns the result in a newly
// created Ciphertext. The input Ciphertext must be of degree two.
func (eval *evaluator) RelinearizeNew(ct0 *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext) {
	ctOut = NewCiphertext(eval.params, 1, ct0.Level())
	eval.Relinearize(ct0, ctOut)
	return
}

// SwitchKeysNew re-encrypts ct0 under a different key and returns the result in a newly created element.
// It requires a SwitchingKey, which is computed from the key under which the Ciphertext is currently encrypted,
// and the key under which the Ciphertext will be re-encrypted.
func (eval *evaluator) SwitchKeysNew(ct0 *rlwe.Ciphertext, switchingKey *rlwe.SwitchingKey) (ctOut *rlwe.Ciphertext) {
	ctOut = NewCiphertext(eval.params, ct0.Degree(), ct0.Level())
	eval.SwitchKeys(ct0, switchingKey, ctOut)
	return
}

// RotateNew rotates the columns of ct0 by k positions to the left, and returns the result in a newly created element.
// If the provided element is a Ciphertext, a key-switching operation is necessary and a rotation key for the specific rotation needs to be provided.
func (eval *evaluator) RotateNew(ct0 *rlwe.Ciphertext, k int) (ctOut *rlwe.Ciphertext) {
	ctOut = NewCiphertext(eval.params, ct0.Degree(), ct0.Level())
	eval.Rotate(ct0, k, ctOut)
	return
}

// Rotate rotates the columns of ct0 by k positions to the left and returns the result in ctOut.
// If the provided element is a Ciphertext, a key-switching operation is necessary and a rotation key for the specific rotation needs to be provided.
func (eval *evaluator) Rotate(ct0 *rlwe.Ciphertext, k int, ctOut *rlwe.Ciphertext) {
	eval.Automorphism(ct0, eval.params.GaloisElementForColumnRotationBy(k), ctOut)
}

// ConjugateNew conjugates ct0 (which is equivalent to a row rotation) and returns the result in a newly
// created element. If the provided element is a Ciphertext, a key-switching operation is necessary and a rotation key
// for the row rotation needs to be provided.
func (eval *evaluator) ConjugateNew(ct0 *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext) {

	if eval.params.RingType() == ring.ConjugateInvariant {
		panic("cannot ConjugateNew: method is not supported when params.RingType() == ring.ConjugateInvariant")
	}

	ctOut = NewCiphertext(eval.params, ct0.Degree(), ct0.Level())
	eval.Conjugate(ct0, ctOut)
	return
}

// Conjugate conjugates ct0 (which is equivalent to a row rotation) and returns the result in ctOut.
// If the provided element is a Ciphertext, a key-switching operation is necessary and a rotation key for the row rotation needs to be provided.
func (eval *evaluator) Conjugate(ct0 *rlwe.Ciphertext, ctOut *rlwe.Ciphertext) {

	if eval.params.RingType() == ring.ConjugateInvariant {
		panic("cannot Conjugate: method is not supported when params.RingType() == ring.ConjugateInvariant")
	}

	eval.Automorphism(ct0, eval.params.GaloisElementForRowRotation(), ctOut)
}

func (eval *evaluator) RotateHoistedNoModDownNew(level int, rotations []int, c0 *ring.Poly, c2DecompQP []ringqp.Poly) (cOut map[int]rlwe.CiphertextQP) {
	ringQ := eval.params.RingQ()
	ringP := eval.params.RingP()
	cOut = make(map[int]rlwe.CiphertextQP)
	for _, i := range rotations {
		if i != 0 {
			cOut[i] = rlwe.CiphertextQP{Value: [2]ringqp.Poly{{Q: ringQ.NewPolyLvl(level), P: ringP.NewPoly()}, {Q: ringQ.NewPolyLvl(level), P: ringP.NewPoly()}}, MetaData: rlwe.MetaData{IsNTT: true}}
			eval.AutomorphismHoistedNoModDown(level, c0, c2DecompQP, eval.params.GaloisElementForColumnRotationBy(i), cOut[i])
		}
	}

	return
}

// ShallowCopy creates a shallow copy of this evaluator in which all the read-only data-structures are
// shared with the receiver and the temporary buffers are reallocated. The receiver and the returned
// Evaluators can be used concurrently.
func (eval *evaluator) ShallowCopy() Evaluator {
	return &evaluator{
		evaluatorBase:    eval.evaluatorBase,
		Evaluator:        eval.Evaluator.ShallowCopy(),
		evaluatorBuffers: newEvaluatorBuffers(eval.evaluatorBase),
	}
}

// WithKey creates a shallow copy of the receiver Evaluator for which the new EvaluationKey is evaluationKey
// and where the temporary buffers are shared. The receiver and the returned Evaluators cannot be used concurrently.
func (eval *evaluator) WithKey(evaluationKey rlwe.EvaluationKey) Evaluator {
	return &evaluator{
		Evaluator:        eval.Evaluator.WithKey(&evaluationKey),
		evaluatorBase:    eval.evaluatorBase,
		evaluatorBuffers: eval.evaluatorBuffers,
	}
}

// Hoisted Rotation combining ModDown and Rescale, One should be clear what he/she is doing with this function.
// Extra Rescale condition must be checked before invoking this function since it is not responsible for checking
// the necessaty of rescaling, but straightly applies one. Also, rotation with step zero is not allowed.
func (eval *evaluator) RotateHoistedModDownAndRescaleOne(ctIn *rlwe.Ciphertext, rotation int, ctOut *rlwe.Ciphertext) {

	if ctIn.Degree() != 1 || ctOut.Degree() != 1 {
		panic("cannot apply Automorphism: input and output Ciphertext must be of degree 1")
	}

	if ctOut.Level() < ctIn.Level()-1 {
		panic("cannot Rescale: not valid ctOut due to too small level budget")
	}

	ringQ := eval.params.RingQ()

	level := ctIn.Level()

	if ctIn.Scale.Cmp(rlwe.NewScale(ringQ.Modulus[level])) == -1 {
		panic("cannot Rescale: ciphertext scale is already smaller than a modulus")
	}
	if ctIn.Level() == 0 {
		panic("cannot Rescale: input Ciphertext already at level 0")
	}

	ctOut.MetaData = ctIn.MetaData

	galEl := eval.params.GaloisElementForColumnRotationBy(rotation)

	if galEl == 1 {
		/*
			if ctOut != ctIn {
				ctOut.Copy(ctIn)
			}
			return
		*/
		panic("meaningless operation: rotataion with step 0")
	}

	// V1 Design:
	eval.DecomposeNTT(level, eval.params.PCount()-1, eval.params.PCount(), ctIn.Value[1], ctIn.IsNTT, eval.BuffDecompQP)

	eval.AutomorphismHoistedModDownAndRescale(level, ctIn.Value[0], eval.BuffDecompQP, galEl, ctOut)
	// V1 Design
	ctOut.Resize(ctOut.Degree(), level-1)
	ctOut.Scale = ctOut.Scale.Div(rlwe.NewScale(ringQ.Modulus[level]))
}

func (eval *evaluator) RotateHoistedAndRescaleOne(ctIn *rlwe.Ciphertext, rotation int, ctOut *rlwe.Ciphertext) {
	eval.AutomorphismHoistedAndRescale(ctIn, eval.params.GaloisElementForColumnRotationBy(rotation), ctOut)

}

// Perform Mult and Sum, user should ensure that ciphertexts and plaintexts should have equal level and scale.
func (eval *evaluator) MulAndSum(ctIn []*rlwe.Ciphertext, ptIn []*rlwe.Plaintext, ctOut *rlwe.Ciphertext) {
	if len(ctIn) == 0 {
		panic("cannot MulAndSum: invalid input length")
	}
	if len(ctIn) != len(ptIn) {
		panic("cannot MulAndSum: unequal input length")
	}
	Minlevel := utils.MinInt(ctIn[0].Level(), ctOut.Level())
	/*
		for i := 1; i < len(ctIn); i++ {
			Minlevel = utils.MinInt(Minlevel, ctIn[i].Level())
		}
		for i := range ptIn {
			Minlevel = utils.MinInt(Minlevel, ptIn[i].Level())
		}
	*/
	ctOut.MetaData = ctIn[0].MetaData
	ctOut.Scale = ctIn[0].Scale.Mul(ptIn[0].Scale)

	ringQ := eval.params.RingQ()

	ctOut.Resize(ctOut.Degree(), Minlevel)

	QiOverF := eval.params.QiOverflowMargin(Minlevel) >> 1

	var cnt int
	for i, ct := range ctIn {
		pt := ptIn[i]
		if cnt == 0 {
			ringQ.MulCoeffsMontgomeryConstantLvl(Minlevel, ct.Value[0], pt.Value, ctOut.Value[0])
			ringQ.MulCoeffsMontgomeryConstantLvl(Minlevel, ct.Value[1], pt.Value, ctOut.Value[1])
			// MulCoeffsMontgomeryConstantLvl
		} else {
			ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(Minlevel, ct.Value[0], pt.Value, ctOut.Value[0])
			ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(Minlevel, ct.Value[1], pt.Value, ctOut.Value[1])
		}

		if cnt%QiOverF == QiOverF-1 {
			ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
			ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
		}
		cnt++
	}
	if cnt%QiOverF != 0 {
		ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
		ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
	}
}

func (eval *evaluator) MultiGroupNetworkTopDownNew(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext) {

	var _a = 0
	var _c = 1

	var _aa = 0
	var _ac = 1
	var _ca = 2
	var _cc = 3

	// create structures to stores the ciphertext.
	pre := make([][]map[int]*rlwe.Ciphertext, len(RRot))
	nex := make([][]map[int]*rlwe.Ciphertext, len(RRot))
	// Rescord which "nex" is acuatally generated:
	nex_cnt := make([][]map[int]int, len(RRot))
	for i := range pre {
		pre[i] = make([]map[int]*rlwe.Ciphertext, 2)
		nex[i] = make([]map[int]*rlwe.Ciphertext, 2)
		nex_cnt[i] = make([]map[int]int, 2)
		for j := range pre[i] {
			pre[i][j] = make(map[int]*rlwe.Ciphertext)
			nex[i][j] = make(map[int]*rlwe.Ciphertext)
			nex_cnt[i][j] = make(map[int]int)
		}
	}

	// Initialize it with input ciphertext.
	pre[0][_a][0] = ctIn.CopyNew()
	pre[0][_c][0] = ctIn
	// nex[0][_a][0] = rlwe.NewCiphertext(eval.Parameters(), ctIn.Degree(), ctIn.Level())

	// Get ring to perform lazy Mult and Add
	ringQ := eval.params.RingQ()
	Maxlevel := ctIn.Level()

	// Pre allocate ctOut:
	// ctOut = NewCiphertext(eval.params, 1, Maxlevel-lv_end+lv_begin+1)

	// Temp Buffers:
	var ctTmp1 *rlwe.Ciphertext
	var ctTmp2 *rlwe.Ciphertext
	var _x, _y, s, t int
	var ecd_level, QiOverFMul, cnt int
	var visitOrder = [4]int{_ac, _aa, _ca, _cc}
	var minScale, divScale rlwe.Scale

	for lv := lv_begin; lv <= lv_end; lv += inc {
		if lv < lv_end {
			// Get encoding level of this round (after rotation's rescale)
			ecd_level = Maxlevel - lv + lv_begin
			// Get Maximum NUmber of Addition before Overflow:
			QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
		}

		// Rotation(Combined with Rescale):
		for m := range RRot {
			for k := range pre[m][_a] {
				if step, exists := RRot[m][lv]; exists {
					ctTmp1 = pre[m][_a][k]

					minScale = eval.params.DefaultScale().Div(rlwe.NewScale(2))
					divScale = pre[m][_a][k].Scale.Div(rlwe.NewScale(eval.params.RingQ().Modulus[pre[m][_a][k].Level()]))

					if lv == lv_begin || divScale.Cmp(minScale) == -1 {
						eval.Rotate(ctTmp1, step, ctTmp1)
					} else {
						if step == 0 {
							eval.Rescale(ctTmp1, eval.params.DefaultScale(), ctTmp1)
						}
						eval.RotateHoistedAndRescaleOne(ctTmp1, step, ctTmp1)
					}

					// Only for debug:
					/*
						ct := pre[m][_a][k]
						fmt.Printf("group %d, level %d nex[%d][%d] with Scale %d, Modlv %d (after rotation)\n", m, lv, _a, k, int(math.Log2(ct.Scale.Float64())), ct.Level())
						decryptor := NewDecryptor(eval.params, sk)
						encoder := NewEncoder(eval.params)
						pt := decryptor.DecryptNew(ct)
						values := encoder.Decode(pt, eval.params.LogSlots())
						realpart := make([]float64, len(values))
						for i := range values {
							realpart[i] = real(values[i])
						}
						fmt.Printf("[")
						for i := 0; i < len(realpart); i++ {
							fmt.Printf("%.0f ", realpart[i])
						}
						fmt.Printf("]\n")
						fmt.Printf("\n")
					*/
				}
			}
			i_g := m
			if len(MMaskpt[m][i_g]) <= lv || MMaskpt[m][i_g][lv] == nil {
				continue
			}
			// handle mapping:
			for _, cls := range visitOrder {

				for k, v := range MMaskpt[m][i_g][lv][cls] {
					_x = cls / 2
					_y = cls % 2
					fmt.Sscanf(k, "%d:%d", &s, &t)

					ctTmp1 = pre[m][_x][s]

					switch {
					case (cls == _aa || cls == _ca):
						if nex[i_g][_y][t] == nil {
							nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ctTmp1.Level())
						}
						ctTmp2 = nex[i_g][_y][t]
						if nex_cnt[i_g][_y][t] == 0 {
							// nex_cnt[i_g][_y][t] = 0
							ctTmp2.MetaData = ctTmp1.MetaData
							ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
							ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
						} else {
							ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
							ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
						}
						if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
							ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
							ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
						}
						nex_cnt[i_g][_y][t]++

					case (cls == _ac && lv == lv_end-1) || (cls == _cc && lv == lv_end-1):
						if nex[i_g][_y][t] == nil {
							nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ecd_level)
						}
						ctTmp2 = nex[i_g][_y][t]
						ctTmp2.MetaData = ctTmp1.MetaData
						ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
						ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
						ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
						ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
						if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
							ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
							ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
						}
						nex_cnt[i_g][_y][t]++

					case (cls == _ac):
						if lv < lv_end-1 {
							nex[i_g][_y][t] = ctTmp1.CopyNew()
						} else if lv == lv_end {
							nex[i_g][_x][t] = ctTmp1 // we only do this once, so no worry for modification.
						}

					case (cls == _cc):
						if lv < lv_end-1 {
							nex[i_g][_y][t] = ctTmp1 // we only do this once per ciphertext, so no worry for modification.
						} else if lv == lv_end {
							if cnt == 0 {
								// nex_cnt[i_g][_y][t] = 0
								ctOut = ctTmp1.CopyNew()
							} else {
								ctTmp2 = ctOut
								// debug
								/*
									if ctTmp1.Level() != ctTmp2.Level() {
										panic("???")
									}
								*/
								ringQ.AddNoMod(ctTmp1.Value[0], ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.AddNoMod(ctTmp1.Value[1], ctTmp2.Value[1], ctTmp2.Value[1])
							}
							if cnt%QiOverFMul == QiOverFMul-1 {
								ctTmp2 = ctOut
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							cnt++
						}
					}
				}

			}
		}

		// Rotation(Combined with Rescale):
		for m := range RRot {
			// Mask on ciphertext:
			for i_g := m + 1; i_g < len(MMaskpt[m]); i_g++ {
				// skip empty ones
				if len(MMaskpt[m][i_g]) <= lv || MMaskpt[m][i_g][lv] == nil {
					continue
				}
				// handle mapping:
				for _, cls := range visitOrder {
					for k, v := range MMaskpt[m][i_g][lv][cls] {
						_x = cls / 2
						_y = cls % 2
						fmt.Sscanf(k, "%d:%d", &s, &t)

						ctTmp1 = pre[m][_x][s]

						switch {
						case (cls == _aa || cls == _ca):
							if nex[i_g][_y][t] == nil {
								nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ctTmp1.Level())
							}
							ctTmp2 = nex[i_g][_y][t]
							if nex_cnt[i_g][_y][t] == 0 {
								// nex_cnt[i_g][_y][t] = 0
								ctTmp2.MetaData = ctTmp1.MetaData
								ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
								ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							} else {
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							}
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++
						}
					}
				}
			}

		}
		if lv < lv_end {

			for m := range nex_cnt {
				for k := range nex_cnt[m] {
					for t, cnt_tmp := range nex_cnt[m][k] {
						if cnt_tmp%QiOverFMul != 0 {
							ctTmp2 = nex[m][k][t]
							ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
							ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
						}
						delete(nex_cnt[m][k], t)
					}
				}
			}

			pre = nex
		} else {
			if cnt%QiOverFMul != 0 {
				ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
				ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
			}

		}

		// only for debug:
		/*
			for i_g := range nex {
				for s := range nex[i_g] {
					nex_arr := make([]int, 0)
					for t := range nex[i_g][s] {
						nex_arr = append(nex_arr, t)
					}
					sort.Slice(nex_arr, func(i, j int) bool {
						return nex_arr[i] < nex_arr[j]
					})
					for _, t := range nex_arr {
						ct := nex[i_g][s][t]
						fmt.Printf("group %d, level %d nex[%d][%d] with Scale %d, Modlv %d\n", i_g, lv, s, t, int(math.Log2(ct.Scale.Float64())), ct.Level())
						decryptor := NewDecryptor(eval.params, sk)
						encoder := NewEncoder(eval.params)
						pt := decryptor.DecryptNew(ct)
						values := encoder.Decode(pt, eval.params.LogSlots())
						realpart := make([]float64, len(values))
						for i := range values {
							realpart[i] = real(values[i])
						}
						fmt.Printf("[")
						for i := 0; i < len(realpart); i++ {
							fmt.Printf("%.0f ", realpart[i])
						}
						fmt.Printf("]\n")
						fmt.Printf("\n")
					}
				}
			}
		*/
	}

	cnt = 0
	ecd_level = Maxlevel - lv_end + lv_begin
	QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
	eval.Rescale(ctOut, eval.params.DefaultScale(), ctOut)
	for m := range pre {
		if len(MMaskpt[m][m][lv_end][_ac]) != 1 {
			continue
		}
		ctTmp1 = pre[m][_a][0]
		// debug
		/*
			if ctTmp1.Level() != ecd_level {
				panic("????")
			}
		*/
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[0], ctTmp1.Value[0], ctOut.Value[0])
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[1], ctTmp1.Value[1], ctOut.Value[1])
		if cnt%QiOverFMul == QiOverFMul-1 {
			ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
			ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
		}
		cnt++
	}
	if cnt%QiOverFMul != 0 {
		ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
		ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
	}

	return
}

func (eval *evaluator) MultiGroupNetworkTopDownNewV2(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext) {
	// stable version.
	var _a = 0
	var _c = 1

	var _aa = 0
	var _ac = 1
	var _ca = 2
	var _cc = 3

	// create structures to stores the ciphertext.
	pre := make([][]map[int]*rlwe.Ciphertext, len(RRot))
	nex := make([][]map[int]*rlwe.Ciphertext, len(RRot))
	// Rescord which "nex" is acuatally generated:
	nex_cnt := make([][]map[int]int, len(RRot))
	for i := range pre {
		pre[i] = make([]map[int]*rlwe.Ciphertext, 2)
		nex[i] = make([]map[int]*rlwe.Ciphertext, 2)
		nex_cnt[i] = make([]map[int]int, 2)
		for j := range pre[i] {
			pre[i][j] = make(map[int]*rlwe.Ciphertext)
			nex[i][j] = make(map[int]*rlwe.Ciphertext)
			nex_cnt[i][j] = make(map[int]int)
		}
	}

	// Initialize it with input ciphertext.
	pre[0][_a][0] = ctIn.CopyNew()
	pre[0][_c][0] = ctIn
	// nex[0][_a][0] = rlwe.NewCiphertext(eval.Parameters(), ctIn.Degree(), ctIn.Level())

	// Get ring to perform lazy Mult and Add
	ringQ := eval.params.RingQ()
	Maxlevel := ctIn.Level()

	// Pre allocate ctOut:
	// ctOut = NewCiphertext(eval.params, 1, Maxlevel-lv_end+lv_begin+1)

	// Temp Buffers:
	var ctTmp1 *rlwe.Ciphertext
	var ctTmp2 *rlwe.Ciphertext
	var _x, _y, s, t int
	var ecd_level, QiOverFMul, cnt int
	var lv_end_rlt int
	var visitOrder = [4]int{_ac, _aa, _ca, _cc}
	var minScale, divScale rlwe.Scale

	for lv := lv_begin; lv <= lv_end; lv += inc {

		/*
			if lv < lv_end {
				// Get encoding level of this round (after rotation's rescale)
				ecd_level = Maxlevel - lv + lv_begin
				// Get Maximum NUmber of Addition before Overflow:
				QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
			}
		*/

		for m := range RRot {

			for k := range pre[m][_a] {
				if step, exists := RRot[m][lv]; exists {
					ctTmp1 = pre[m][_a][k]

					minScale = eval.params.DefaultScale().Div(rlwe.NewScale(2))
					divScale = pre[m][_a][k].Scale.Div(rlwe.NewScale(eval.params.RingQ().Modulus[pre[m][_a][k].Level()]))

					if lv == lv_begin || divScale.Cmp(minScale) == -1 {
						eval.Rotate(ctTmp1, step, ctTmp1)
					} else {
						if step == 0 {
							eval.Rescale(ctTmp1, eval.params.DefaultScale(), ctTmp1)
						}
						eval.RotateHoistedAndRescaleOne(ctTmp1, step, ctTmp1)
					}

					// Only for debug:
					/*
						ct := pre[m][_a][k]
						fmt.Printf("group %d, level %d nex[%d][%d] with Scale %d, Modlv %d (after rotation)\n", m, lv, _a, k, int(math.Log2(ct.Scale.Float64())), ct.Level())
						decryptor := NewDecryptor(eval.params, sk)
						encoder := NewEncoder(eval.params)
						pt := decryptor.DecryptNew(ct)
						values := encoder.Decode(pt, eval.params.LogSlots())
						realpart := make([]float64, len(values))
						for i := range values {
							realpart[i] = real(values[i])
						}
						fmt.Printf("[")
						for i := 0; i < len(realpart); i++ {
							fmt.Printf("%.0f ", realpart[i])
						}
						fmt.Printf("]\n")
						fmt.Printf("\n")
					*/
				}
			}
		}

		// Rotation(Combined with Rescale):
		for i_g := len(RRot) - 1; i_g >= 0; i_g-- {

			// 0903:
			lv_end_rlt = len(MMaskpt[i_g][i_g]) - 1

			if lv < lv_end_rlt {
				// Get encoding level of this round (after rotation's rescale)
				ecd_level = Maxlevel - lv + lv_begin - (lv_end - lv_end_rlt)
				// Get Maximum NUmber of Addition before Overflow:
				QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
			}

			for m := i_g; m >= 0; m-- {
				if len(MMaskpt[m][i_g]) <= lv || MMaskpt[m][i_g][lv] == nil {
					continue
				}
				// handle mapping:
				for _, cls := range visitOrder {

					for k, v := range MMaskpt[m][i_g][lv][cls] {
						_x = cls / 2
						_y = cls % 2
						fmt.Sscanf(k, "%d:%d", &s, &t)

						ctTmp1 = pre[m][_x][s]

						switch {
						case (cls == _aa || cls == _ca):
							if nex[i_g][_y][t] == nil {
								nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ctTmp1.Level())
							}
							ctTmp2 = nex[i_g][_y][t]
							if nex_cnt[i_g][_y][t] == 0 {
								// nex_cnt[i_g][_y][t] = 0
								ctTmp2.MetaData = ctTmp1.MetaData
								ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
								ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							} else {
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							}
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac && lv == lv_end_rlt-1) || (cls == _cc && lv == lv_end_rlt-1):
							if nex[i_g][_y][t] == nil {
								nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ecd_level)
							}
							ctTmp2 = nex[i_g][_y][t]
							ctTmp2.MetaData = ctTmp1.MetaData
							ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
							ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac):
							if lv < lv_end_rlt-1 {
								nex[i_g][_y][t] = ctTmp1.CopyNew()
							} else if lv == lv_end_rlt {
								nex[i_g][_x][t] = ctTmp1 // we only do this once, so no worry for modification.
							}

						case (cls == _cc):
							if lv < lv_end_rlt-1 {
								nex[i_g][_y][t] = ctTmp1 // we only do this once per ciphertext, so no worry for modification.
							} else if lv == lv_end_rlt {
								if cnt == 0 {
									// nex_cnt[i_g][_y][t] = 0
									ctOut = ctTmp1.CopyNew()
								} else {
									ctTmp2 = ctOut
									// debug
									/*
										if ctTmp1.Level() != ctTmp2.Level() {
											panic("???")
										}
									*/
									ringQ.AddNoMod(ctTmp1.Value[0], ctTmp2.Value[0], ctTmp2.Value[0])
									ringQ.AddNoMod(ctTmp1.Value[1], ctTmp2.Value[1], ctTmp2.Value[1])
								}
								if cnt%QiOverFMul == QiOverFMul-1 {
									ctTmp2 = ctOut
									ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
									ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
								}
								cnt++
							}
						}
					}

				}
			}

			if lv < lv_end_rlt {
				m := i_g
				for k := range nex_cnt[m] {
					for t, cnt_tmp := range nex_cnt[m][k] {
						if cnt_tmp%QiOverFMul != 0 {
							ctTmp2 = nex[m][k][t]
							ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
							ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
						}
						delete(nex_cnt[m][k], t)
					}
				}
				pre[m] = nex[m]
			}
		}

		// only for debug:
		/*
			for i_g := range nex {
				for s := range nex[i_g] {
					nex_arr := make([]int, 0)
					for t := range nex[i_g][s] {
						nex_arr = append(nex_arr, t)
					}
					sort.Slice(nex_arr, func(i, j int) bool {
						return nex_arr[i] < nex_arr[j]
					})
					for _, t := range nex_arr {
						ct := nex[i_g][s][t]
						fmt.Printf("group %d, level %d nex[%d][%d] with Scale %d, Modlv %d\n", i_g, lv, s, t, int(math.Log2(ct.Scale.Float64())), ct.Level())
						decryptor := NewDecryptor(eval.params, sk)
						encoder := NewEncoder(eval.params)
						pt := decryptor.DecryptNew(ct)
						values := encoder.Decode(pt, eval.params.LogSlots())
						realpart := make([]float64, len(values))
						for i := range values {
							realpart[i] = real(values[i])
						}
						fmt.Printf("[")
						for i := 0; i < len(realpart); i++ {
							fmt.Printf("%.0f ", realpart[i])
						}
						fmt.Printf("]\n")
						fmt.Printf("\n")
					}
				}
			}
		*/
	}
	ecd_level = Maxlevel - (lv_end - 1) + lv_begin
	QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
	if cnt%QiOverFMul != 0 {
		ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
		ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
	}
	eval.Rescale(ctOut, eval.params.DefaultScale(), ctOut)

	var last_lv int
	cnt = 0
	ecd_level = Maxlevel - lv_end + lv_begin
	QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
	for m := range pre {
		last_lv = (len(MMaskpt[m][m]) - 1) - 1
		if len(MMaskpt[m][m][last_lv][_aa]) != 1 {
			continue
		}
		ctTmp1 = pre[m][_a][0]
		// debug
		/*
			if ctTmp1.Level() != ecd_level {
				panic("????")
			}
		*/
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[0], ctTmp1.Value[0], ctOut.Value[0])
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[1], ctTmp1.Value[1], ctOut.Value[1])
		if cnt%QiOverFMul == QiOverFMul-1 {
			ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
			ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
		}
		cnt++
	}
	if cnt%QiOverFMul != 0 {
		ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
		ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
	}

	return
}

func (eval *evaluator) MultiGroupNetworkTopDownNewV3(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, pre, nex [][]map[int]*rlwe.Ciphertext, nex_cnt [][]map[int]int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext) {
	// memory pre-allocate version of V2.
	var _a = 0
	var _c = 1

	var _aa = 0
	var _ac = 1
	var _ca = 2
	var _cc = 3

	// create structures to stores the ciphertext.
	/*
		pre := make([][]map[int]*rlwe.Ciphertext, len(RRot))
		nex := make([][]map[int]*rlwe.Ciphertext, len(RRot))
		// Rescord which "nex" is acuatally generated:
		nex_cnt := make([][]map[int]int, len(RRot))
		for i := range pre {
			pre[i] = make([]map[int]*rlwe.Ciphertext, 2)
			nex[i] = make([]map[int]*rlwe.Ciphertext, 2)
			nex_cnt[i] = make([]map[int]int, 2)
			for j := range pre[i] {
				pre[i][j] = make(map[int]*rlwe.Ciphertext)
				nex[i][j] = make(map[int]*rlwe.Ciphertext)
				nex_cnt[i][j] = make(map[int]int)
			}
		}
	*/

	// Initialize it with input ciphertext.
	pre[0][_a][0].Copy(ctIn)
	pre[0][_c][0] = ctIn

	// nex[0][_a][0] = rlwe.NewCiphertext(eval.Parameters(), ctIn.Degree(), ctIn.Level())

	// Get ring to perform lazy Mult and Add
	ringQ := eval.params.RingQ()
	Maxlevel := ctIn.Level()

	// Pre allocate ctOut:
	// ctOut = NewCiphertext(eval.params, 1, Maxlevel-lv_end+lv_begin+1)

	// Temp Buffers:
	var ctTmp1 *rlwe.Ciphertext
	var ctTmp2 *rlwe.Ciphertext
	var _x, _y, s, t int
	var ecd_level, QiOverFMul, cnt int
	var lv_end_rlt int
	var visitOrder = [4]int{_ac, _aa, _ca, _cc}
	var minScale, divScale rlwe.Scale

	var elapsed time.Duration

	for lv := lv_begin; lv <= lv_end; lv += inc {

		now := time.Now()

		/*
			if lv < lv_end {
				// Get encoding level of this round (after rotation's rescale)
				ecd_level = Maxlevel - lv + lv_begin
				// Get Maximum NUmber of Addition before Overflow:
				QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
			}
		*/

		for m := range RRot {

			for k := range pre[m][_a] {
				if step, exists := RRot[m][lv]; exists {
					ctTmp1 = pre[m][_a][k]

					minScale = eval.params.DefaultScale().Div(rlwe.NewScale(2))
					divScale = pre[m][_a][k].Scale.Div(rlwe.NewScale(eval.params.RingQ().Modulus[pre[m][_a][k].Level()]))

					if lv == lv_begin || divScale.Cmp(minScale) == -1 {
						eval.Rotate(ctTmp1, step, ctTmp1)
					} else {
						if step == 0 {
							eval.Rescale(ctTmp1, eval.params.DefaultScale(), ctTmp1)
						}
						eval.RotateHoistedAndRescaleOne(ctTmp1, step, ctTmp1)
					}

					// Only for debug:
					/*
						ct := pre[m][_a][k]
						fmt.Printf("group %d, level %d nex[%d][%d] with Scale %d, Modlv %d (after rotation)\n", m, lv, _a, k, int(math.Log2(ct.Scale.Float64())), ct.Level())
						decryptor := NewDecryptor(eval.params, sk)
						encoder := NewEncoder(eval.params)
						pt := decryptor.DecryptNew(ct)
						values := encoder.Decode(pt, eval.params.LogSlots())
						realpart := make([]float64, len(values))
						for i := range values {
							realpart[i] = real(values[i])
						}
						fmt.Printf("[")
						for i := 0; i < len(realpart); i++ {
							fmt.Printf("%.0f ", realpart[i])
						}
						fmt.Printf("]\n")
						fmt.Printf("\n")
					*/
				}
			}
		}

		// Rotation(Combined with Rescale):
		for i_g := len(RRot) - 1; i_g >= 0; i_g-- {

			// 0903:
			lv_end_rlt = len(MMaskpt[i_g][i_g]) - 1

			if lv < lv_end_rlt {
				// Get encoding level of this round (after rotation's rescale)
				ecd_level = Maxlevel - lv + lv_begin - (lv_end - lv_end_rlt)
				// Get Maximum NUmber of Addition before Overflow:
				QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
			}

			for m := i_g; m >= 0; m-- {
				if len(MMaskpt[m][i_g]) <= lv || MMaskpt[m][i_g][lv] == nil {
					continue
				}
				// handle mapping:
				for _, cls := range visitOrder {

					for k, v := range MMaskpt[m][i_g][lv][cls] {
						_x = cls / 2
						_y = cls % 2
						fmt.Sscanf(k, "%d:%d", &s, &t)

						ctTmp1 = pre[m][_x][s]

						switch {
						case (cls == _aa || cls == _ca):
							/*
								if nex[i_g][_y][t] == nil {
									nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ctTmp1.Level())
								}
							*/
							ctTmp2 = nex[i_g][_y][t]
							if nex_cnt[i_g][_y][t] == 0 {
								// nex_cnt[i_g][_y][t] = 0
								ctTmp2.MetaData = ctTmp1.MetaData
								ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
								ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							} else {
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							}
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac && lv == lv_end_rlt-1) || (cls == _cc && lv == lv_end_rlt-1):
							/*
								if nex[i_g][_y][t] == nil {
									nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ecd_level)
								}
							*/
							ctTmp2 = nex[i_g][_y][t]
							ctTmp2.MetaData = ctTmp1.MetaData
							ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
							ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac):
							if lv < lv_end_rlt-1 {
								nex[i_g][_y][t].Copy(ctTmp1) // = ctTmp1.CopyNew()
							} else if lv == lv_end_rlt {
								nex[i_g][_x][t] = ctTmp1 // we only do this once, so no worry for modification.
							}

						case (cls == _cc):
							if lv < lv_end_rlt-1 {
								nex[i_g][_y][t] = ctTmp1 // we only do this once per ciphertext, so no worry for modification.
							} else if lv == lv_end_rlt {
								if cnt == 0 {
									// nex_cnt[i_g][_y][t] = 0
									ctOut = ctTmp1.CopyNew()
								} else {
									ctTmp2 = ctOut
									// debug
									/*
										if ctTmp1.Level() != ctTmp2.Level() {
											panic("???")
										}
									*/
									ringQ.AddNoMod(ctTmp1.Value[0], ctTmp2.Value[0], ctTmp2.Value[0])
									ringQ.AddNoMod(ctTmp1.Value[1], ctTmp2.Value[1], ctTmp2.Value[1])
								}
								if cnt%QiOverFMul == QiOverFMul-1 {
									ctTmp2 = ctOut
									ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
									ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
								}
								cnt++
							}
						}
					}

				}
			}

			if lv < lv_end_rlt {
				m := i_g
				for k := range nex_cnt[m] {
					for t, cnt_tmp := range nex_cnt[m][k] {
						if cnt_tmp%QiOverFMul != 0 {
							ctTmp2 = nex[m][k][t]
							ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
							ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
						}
						delete(nex_cnt[m][k], t)
					}
				}
				pre[m] = nex[m]
			}
		}

		// only for debug:
		/*
			for i_g := range nex {
				for s := range nex[i_g] {
					nex_arr := make([]int, 0)
					for t := range nex[i_g][s] {
						nex_arr = append(nex_arr, t)
					}
					sort.Slice(nex_arr, func(i, j int) bool {
						return nex_arr[i] < nex_arr[j]
					})
					for _, t := range nex_arr {
						ct := nex[i_g][s][t]
						fmt.Printf("group %d, level %d nex[%d][%d] with Scale %d, Modlv %d\n", i_g, lv, s, t, int(math.Log2(ct.Scale.Float64())), ct.Level())
						decryptor := NewDecryptor(eval.params, sk)
						encoder := NewEncoder(eval.params)
						pt := decryptor.DecryptNew(ct)
						values := encoder.Decode(pt, eval.params.LogSlots())
						realpart := make([]float64, len(values))
						for i := range values {
							realpart[i] = real(values[i])
						}
						fmt.Printf("[")
						for i := 0; i < len(realpart); i++ {
							fmt.Printf("%.0f ", realpart[i])
						}
						fmt.Printf("]\n")
						fmt.Printf("\n")
					}
				}
			}
		*/

		elapsed_tmp := time.Since(now)
		fmt.Printf("lv %d consume %s \n", lv, elapsed_tmp)
		elapsed += elapsed_tmp

	}

	fmt.Printf("Total lvs consume %s \n", elapsed)

	ecd_level = Maxlevel - (lv_end - 1) + lv_begin
	QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
	if cnt%QiOverFMul != 0 {
		ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
		ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
	}
	eval.Rescale(ctOut, eval.params.DefaultScale(), ctOut)

	var last_lv int
	cnt = 0
	ecd_level = Maxlevel - lv_end + lv_begin
	QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
	for m := range nex { // for m := range pre { // 10.06.2025
		last_lv = (len(MMaskpt[m][m]) - 1) - 1
		_ = last_lv

		/*
			if m >= 1 { // added 10.06.2025
				if len(MMaskpt[m][m][last_lv]) == 0 {
					continue
				} else if len(MMaskpt[m][m][last_lv][_aa]) != 1 {
					continue
				}
			}
		*/

		ctTmp1 = nex[m][_a][0] // pre[m][_a][0] // 10.06.2025

		// debug
		/*
			fmt.Printf("nex[%d][%d][%d] with Scale %d, Modlv %d\n", m, _a, 0, int(math.Log2(ctTmp1.Scale.Float64())), ctTmp1.Level())
			decryptor := NewDecryptor(eval.params, sk)
			encoder := NewEncoder(eval.params)
			pt := decryptor.DecryptNew(ctTmp1)
			values := encoder.Decode(pt, eval.params.LogSlots())
			realpart := make([]float64, len(values))
			for i := range values {
				realpart[i] = real(values[i])
			}
			fmt.Printf("[")
			for i := 0; i < len(realpart); i++ {
				fmt.Printf("%.0f ", realpart[i])
			}
			fmt.Printf("]\n")
			fmt.Printf("\n")
		*/
		// debug
		/*
			if ctTmp1.Level() != ecd_level {
				panic("????")
			}
		*/
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[0], ctTmp1.Value[0], ctOut.Value[0])
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[1], ctTmp1.Value[1], ctOut.Value[1])
		if cnt%QiOverFMul == QiOverFMul-1 {
			ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
			ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
		}
		cnt++
	}
	if cnt%QiOverFMul != 0 {
		ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
		ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
	}

	return
}

func (eval *evaluator) MultiGroupNetworkTopDownNewV3_tick(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, pre, nex [][]map[int]*rlwe.Ciphertext, nex_cnt [][]map[int]int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext, seg []time.Duration) {
	// memory pre-allocate version of V2.
	var _a = 0
	var _c = 1

	var _aa = 0
	var _ac = 1
	var _ca = 2
	var _cc = 3

	// create structures to stores the ciphertext.
	/*
		pre := make([][]map[int]*rlwe.Ciphertext, len(RRot))
		nex := make([][]map[int]*rlwe.Ciphertext, len(RRot))
		// Rescord which "nex" is acuatally generated:
		nex_cnt := make([][]map[int]int, len(RRot))
		for i := range pre {
			pre[i] = make([]map[int]*rlwe.Ciphertext, 2)
			nex[i] = make([]map[int]*rlwe.Ciphertext, 2)
			nex_cnt[i] = make([]map[int]int, 2)
			for j := range pre[i] {
				pre[i][j] = make(map[int]*rlwe.Ciphertext)
				nex[i][j] = make(map[int]*rlwe.Ciphertext)
				nex_cnt[i][j] = make(map[int]int)
			}
		}
	*/

	// Initialize it with input ciphertext.
	pre[0][_a][0].Copy(ctIn)
	pre[0][_c][0] = ctIn

	// nex[0][_a][0] = rlwe.NewCiphertext(eval.Parameters(), ctIn.Degree(), ctIn.Level())

	// Get ring to perform lazy Mult and Add
	ringQ := eval.params.RingQ()
	Maxlevel := ctIn.Level()

	// Pre allocate ctOut:
	// ctOut = NewCiphertext(eval.params, 1, Maxlevel-lv_end+lv_begin+1)

	// Temp Buffers:
	var ctTmp1 *rlwe.Ciphertext
	var ctTmp2 *rlwe.Ciphertext
	var _x, _y, s, t int
	var ecd_level, QiOverFMul, cnt int
	var lv_end_rlt int
	var visitOrder = [4]int{_ac, _aa, _ca, _cc}
	var minScale, divScale rlwe.Scale

	seg = make([]time.Duration, 0)
	var elapsed time.Duration

	for lv := lv_begin; lv <= lv_end; lv += inc {

		now := time.Now()

		/*
			if lv < lv_end {
				// Get encoding level of this round (after rotation's rescale)
				ecd_level = Maxlevel - lv + lv_begin
				// Get Maximum NUmber of Addition before Overflow:
				QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
			}
		*/

		for m := range RRot {

			for k := range pre[m][_a] {
				if step, exists := RRot[m][lv]; exists {
					ctTmp1 = pre[m][_a][k]

					minScale = eval.params.DefaultScale().Div(rlwe.NewScale(2))
					divScale = pre[m][_a][k].Scale.Div(rlwe.NewScale(eval.params.RingQ().Modulus[pre[m][_a][k].Level()]))

					if lv == lv_begin || divScale.Cmp(minScale) == -1 {
						eval.Rotate(ctTmp1, step, ctTmp1)
						fmt.Printf("Rotation performed on Level %d\n", ctTmp1.Level())
					} else {
						if step == 0 {
							eval.Rescale(ctTmp1, eval.params.DefaultScale(), ctTmp1)
						}
						fmt.Printf("Rotation+Rescale to perform on Level %d\n", ctTmp1.Level())
						eval.RotateHoistedAndRescaleOne(ctTmp1, step, ctTmp1)

					}

					// Only for debug:
					/*
						ct := pre[m][_a][k]
						fmt.Printf("group %d, level %d nex[%d][%d] with Scale %d, Modlv %d (after rotation)\n", m, lv, _a, k, int(math.Log2(ct.Scale.Float64())), ct.Level())
						decryptor := NewDecryptor(eval.params, sk)
						encoder := NewEncoder(eval.params)
						pt := decryptor.DecryptNew(ct)
						values := encoder.Decode(pt, eval.params.LogSlots())
						realpart := make([]float64, len(values))
						for i := range values {
							realpart[i] = real(values[i])
						}
						fmt.Printf("[")
						for i := 0; i < len(realpart); i++ {
							fmt.Printf("%.0f ", realpart[i])
						}
						fmt.Printf("]\n")
						fmt.Printf("\n")
					*/
				}
			}
		}

		// Rotation(Combined with Rescale):
		for i_g := len(RRot) - 1; i_g >= 0; i_g-- {

			// 0903:
			lv_end_rlt = len(MMaskpt[i_g][i_g]) - 1

			if lv < lv_end_rlt {
				// Get encoding level of this round (after rotation's rescale)
				ecd_level = Maxlevel - lv + lv_begin - (lv_end - lv_end_rlt)
				// Get Maximum NUmber of Addition before Overflow:
				QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
			}

			for m := i_g; m >= 0; m-- {
				if len(MMaskpt[m][i_g]) <= lv || MMaskpt[m][i_g][lv] == nil {
					continue
				}
				// handle mapping:
				for _, cls := range visitOrder {

					for k, v := range MMaskpt[m][i_g][lv][cls] {
						_x = cls / 2
						_y = cls % 2
						fmt.Sscanf(k, "%d:%d", &s, &t)

						ctTmp1 = pre[m][_x][s]

						switch {
						case (cls == _aa || cls == _ca):
							/*
								if nex[i_g][_y][t] == nil {
									nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ctTmp1.Level())
								}
							*/
							ctTmp2 = nex[i_g][_y][t]
							if nex_cnt[i_g][_y][t] == 0 {
								// nex_cnt[i_g][_y][t] = 0
								ctTmp2.MetaData = ctTmp1.MetaData
								ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
								ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							} else {
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							}
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac && lv == lv_end_rlt-1) || (cls == _cc && lv == lv_end_rlt-1):
							/*
								if nex[i_g][_y][t] == nil {
									nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ecd_level)
								}
							*/
							ctTmp2 = nex[i_g][_y][t]
							ctTmp2.MetaData = ctTmp1.MetaData
							ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
							ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac):
							if lv < lv_end_rlt-1 {
								nex[i_g][_y][t].Copy(ctTmp1) // = ctTmp1.CopyNew()
							} else if lv == lv_end_rlt {
								nex[i_g][_x][t] = ctTmp1 // we only do this once, so no worry for modification.
							}

						case (cls == _cc):
							if lv < lv_end_rlt-1 {
								nex[i_g][_y][t] = ctTmp1 // we only do this once per ciphertext, so no worry for modification.
							} else if lv == lv_end_rlt {
								if cnt == 0 {
									// nex_cnt[i_g][_y][t] = 0
									ctOut = ctTmp1.CopyNew()
								} else {
									ctTmp2 = ctOut
									// debug
									/*
										if ctTmp1.Level() != ctTmp2.Level() {
											panic("???")
										}
									*/
									ringQ.AddNoMod(ctTmp1.Value[0], ctTmp2.Value[0], ctTmp2.Value[0])
									ringQ.AddNoMod(ctTmp1.Value[1], ctTmp2.Value[1], ctTmp2.Value[1])
								}
								if cnt%QiOverFMul == QiOverFMul-1 {
									ctTmp2 = ctOut
									ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
									ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
								}
								cnt++
							}
						}
					}

				}
			}

			if lv < lv_end_rlt {
				m := i_g
				for k := range nex_cnt[m] {
					for t, cnt_tmp := range nex_cnt[m][k] {
						if cnt_tmp%QiOverFMul != 0 {
							ctTmp2 = nex[m][k][t]
							ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
							ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
						}
						delete(nex_cnt[m][k], t)
					}
				}
				pre[m] = nex[m]
			}
		}

		// only for debug:
		/*
			for i_g := range nex {
				for s := range nex[i_g] {
					nex_arr := make([]int, 0)
					for t := range nex[i_g][s] {
						nex_arr = append(nex_arr, t)
					}
					sort.Slice(nex_arr, func(i, j int) bool {
						return nex_arr[i] < nex_arr[j]
					})
					for _, t := range nex_arr {
						ct := nex[i_g][s][t]
						fmt.Printf("group %d, level %d nex[%d][%d] with Scale %d, Modlv %d\n", i_g, lv, s, t, int(math.Log2(ct.Scale.Float64())), ct.Level())
						decryptor := NewDecryptor(eval.params, sk)
						encoder := NewEncoder(eval.params)
						pt := decryptor.DecryptNew(ct)
						values := encoder.Decode(pt, eval.params.LogSlots())
						realpart := make([]float64, len(values))
						for i := range values {
							realpart[i] = real(values[i])
						}
						fmt.Printf("[")
						for i := 0; i < len(realpart); i++ {
							fmt.Printf("%.0f ", realpart[i])
						}
						fmt.Printf("]\n")
						fmt.Printf("\n")
					}
				}
			}
		*/

		elapsed_tmp := time.Since(now)
		fmt.Printf("lv %d consume %s \n", lv, elapsed_tmp)
		elapsed += elapsed_tmp
		seg = append(seg, elapsed_tmp)
	}

	fmt.Printf("Total lvs consume %s \n", elapsed)

	ecd_level = Maxlevel - (lv_end - 1) + lv_begin
	QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
	if cnt%QiOverFMul != 0 {
		ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
		ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
	}
	eval.Rescale(ctOut, eval.params.DefaultScale(), ctOut)

	var last_lv int
	cnt = 0
	ecd_level = Maxlevel - lv_end + lv_begin
	QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
	for m := range nex { // for m := range pre { // 10.06.2025
		last_lv = (len(MMaskpt[m][m]) - 1) - 1
		_ = last_lv

		/*
			if m >= 1 { // added 10.06.2025
				if len(MMaskpt[m][m][last_lv]) == 0 {
					continue
				} else if len(MMaskpt[m][m][last_lv][_aa]) != 1 {
					continue
				}
			}
		*/

		ctTmp1 = nex[m][_a][0] // pre[m][_a][0] // 10.06.2025

		// debug
		/*
			fmt.Printf("nex[%d][%d][%d] with Scale %d, Modlv %d\n", m, _a, 0, int(math.Log2(ctTmp1.Scale.Float64())), ctTmp1.Level())
			decryptor := NewDecryptor(eval.params, sk)
			encoder := NewEncoder(eval.params)
			pt := decryptor.DecryptNew(ctTmp1)
			values := encoder.Decode(pt, eval.params.LogSlots())
			realpart := make([]float64, len(values))
			for i := range values {
				realpart[i] = real(values[i])
			}
			fmt.Printf("[")
			for i := 0; i < len(realpart); i++ {
				fmt.Printf("%.0f ", realpart[i])
			}
			fmt.Printf("]\n")
			fmt.Printf("\n")
		*/
		// debug
		/*
			if ctTmp1.Level() != ecd_level {
				panic("????")
			}
		*/
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[0], ctTmp1.Value[0], ctOut.Value[0])
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[1], ctTmp1.Value[1], ctOut.Value[1])
		if cnt%QiOverFMul == QiOverFMul-1 {
			ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
			ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
		}
		cnt++
	}
	if cnt%QiOverFMul != 0 {
		ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
		ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
	}

	return
}

func (eval *evaluator) PrepareNexAndPre(MMaskpt [][][][]map[string]*rlwe.Plaintext) (pre, nex [][]map[int]*rlwe.Ciphertext, nex_cnt [][]map[int]int) {

	var _a = 0
	var _c = 1
	// Compute the maximum number of ciphertexts for each group:
	maxNum := len(MMaskpt[0][0])

	// create structures to stores the ciphertext.
	pre = make([][]map[int]*rlwe.Ciphertext, len(MMaskpt))
	nex = make([][]map[int]*rlwe.Ciphertext, len(MMaskpt))

	// Rescord which "nex" is acuatally generated:
	nex_cnt = make([][]map[int]int, len(MMaskpt))
	for i := range pre {
		pre[i] = make([]map[int]*rlwe.Ciphertext, 2)
		nex[i] = make([]map[int]*rlwe.Ciphertext, 2)
		nex_cnt[i] = make([]map[int]int, 2)

		pre[i][_a] = make(map[int]*rlwe.Ciphertext)
		pre[i][_c] = make(map[int]*rlwe.Ciphertext)
		nex[i][_a] = make(map[int]*rlwe.Ciphertext)
		nex[i][_c] = make(map[int]*rlwe.Ciphertext)
		nex_cnt[i][_a] = make(map[int]int)
		nex_cnt[i][_c] = make(map[int]int)

		pre[i][_a][0] = NewCiphertext(eval.params, 1, eval.params.MaxLevel())
		nex[i][_a][0] = NewCiphertext(eval.params, 1, eval.params.MaxLevel())

		for k := 0; k < maxNum; k++ {
			pre[i][_c][k] = NewCiphertext(eval.params, 1, eval.params.MaxLevel())
			nex[i][_c][k] = NewCiphertext(eval.params, 1, eval.params.MaxLevel())
		}
	}

	return
}

func GenCollpasedLevelsInfoUpper(encoder Encoder, nexcollapse_Pt [][]map[int]map[int][]int, collapse_depth int, level int, scale rlwe.Scale, logslots int) (rslt collapsedlevelsInfo) {
	enc, ok := encoder.(*encoderComplex128)
	if !ok {
		panic("cannot GenLinearTransform: encoder should be an encoderComplex128")
	}
	params := enc.params

	rslt = collapsedlevelsInfo{nexcollapse: nil, nexcollapse_pt: nexcollapse_Pt, totalDiffRots: nil, level: level, collapse_depth: collapse_depth}
	rslt.totalDiffRots = make(map[int]int)
	rslt.nexcollapse = make([][]map[int]map[int]ringqp.Poly, len(nexcollapse_Pt))
	levelQ := level
	levelP := params.PCount() - 1
	for i_g := range nexcollapse_Pt {
		// entering target level of the i_g-th group
		rslt.nexcollapse[i_g] = make([]map[int]map[int]ringqp.Poly, len(nexcollapse_Pt[i_g]))
		for t := range nexcollapse_Pt[i_g] {
			// entering target type of node
			rslt.nexcollapse[i_g][t] = make(map[int]map[int]ringqp.Poly)
			for idx := range nexcollapse_Pt[i_g][t] {
				// entering a specific node:
				rslt.nexcollapse[i_g][t][idx] = make(map[int]ringqp.Poly)
				for m_id := range nexcollapse_Pt[i_g][t][idx] {

					// entering a mask in the node, the mask corresponding to one rotated ciphertext.
					// the m_id is the rotation step of that rotated ciphertext.
					mask_float64 := make([]float64, len(nexcollapse_Pt[i_g][t][idx][m_id]))
					for i := range mask_float64 {
						mask_float64[i] = float64(nexcollapse_Pt[i_g][t][idx][m_id][i])
					}
					rslt.nexcollapse[i_g][t][idx][m_id] = params.RingQP().NewPolyLvl(levelQ, levelP)
					enc.Embed(mask_float64, logslots, scale, true, rslt.nexcollapse[i_g][t][idx][m_id])

					// record the rotation step.
					rslt.totalDiffRots[m_id] = 1
				}
			}
		}
	}

	// We want to combine some of the nodes' mask into one, for those node carring very less values.
	// Plan to create at most 7 combined maps
	// Project currently deferred...
	combinedmap_pt := make([]map[int][]int, 0)
	bound := 14
	// the combination comes from the end to the front.
	for i_g := len(nexcollapse_Pt) - 1; i_g >= 0; i_g-- {
		for t := 1; t >= 0; t-- {
			for idx := range nexcollapse_Pt[i_g][t] {
				// entering a specific node:
				if len(nexcollapse_Pt[i_g][t][idx]) < 1 {
					continue
				}
				// If the combinedmap is empty, then it let occupy one:
				if len(combinedmap_pt) == 0 {
					// fmt.Printf("nexcollpase_Pt[%d][%d][%d] first occupies the %d-th combined map\n", i_g, t, idx, len(combinedmap_pt))
					combinedmap_pt = append(combinedmap_pt, nexcollapse_Pt[i_g][t][idx])
				} else {
					// traverse all existing combined mask, see if we can fit this node
					find_room := false
					for i := 0; i < len(combinedmap_pt); i++ {
						noconflict_all := true
						for m_id, mask := range nexcollapse_Pt[i_g][t][idx] {
							_ = m_id
							// for each mask of the node, traverse all existing masks in the map to see if there are any conflict.
							noconflict := true
							for m_idC, maskC := range combinedmap_pt[i] {
								_ = m_idC
								for e := range maskC {
									if maskC[e] == 1 && mask[e] == 1 {
										noconflict = false
										break
									}
								}
								if !noconflict {
									break
								}
							}
							if !noconflict {
								noconflict_all = false
								break
							}
						}
						if noconflict_all {
							find_room = true
							// no conflict found in combined map, so we place this entry here,
							// fmt.Printf("Merging nexcollpase_Pt[%d][%d][%d] to the %d-th combined map\n", i_g, t, idx, i)
							for m_id, mask := range nexcollapse_Pt[i_g][t][idx] {
								if _, exists := combinedmap_pt[i][m_id]; exists {
									for e := range combinedmap_pt[i][m_id] {
										combinedmap_pt[i][m_id][e] += mask[e]
									}
								} else {
									combinedmap_pt[i][m_id] = mask
								}
							}
							break
						}
					}
					// to this step, it indicates that no room for this node, so we whether check if we have upto 7 nodes
					if len(combinedmap_pt) < bound && (!(find_room)) {
						// fmt.Printf("nexcollpase_Pt[%d][%d][%d] first occupies the %d-th combined map\n", i_g, t, idx, len(combinedmap_pt))
						combinedmap_pt = append(combinedmap_pt, nexcollapse_Pt[i_g][t][idx])
					}
				}
			}
		}
	}

	return
}

func (eval *evaluator) MultiGroupNetworkCollapseUpper(ctIn *rlwe.Ciphertext, collapsed collapsedlevelsInfo, sign int, n int, sk *rlwe.SecretKey, nex [][]map[int]*rlwe.Ciphertext) {
	// perform Upper levels' Collpasing
	// first is to prepare all rotation steps.
	// Restricted to only use rotation keys corresponding to power of 2.
	// The logic to judge whether we need to rotate a power of 2 goes like this:
	// 1. Mod the rots to let them have equal sign.
	totalDiffRots := collapsed.totalDiffRots
	nexcollapse := collapsed.nexcollapse
	totalDiffRots_tmp := make(map[int]int)
	if sign == -1 {
		for dist := range totalDiffRots {
			if dist >= n || dist <= -n {
				panic("|invalid dist| >= n")
			} else if dist > 0 {
				totalDiffRots_tmp[dist-n] = 1
			} else {
				totalDiffRots_tmp[dist] = 1
			}
		}
	} else {
		for dist := range totalDiffRots {
			if dist >= n || dist <= -n {
				panic("|invalid dist| >= n")
			} else if dist < 0 {
				totalDiffRots_tmp[dist+n] = 1
			} else {
				totalDiffRots_tmp[dist] = 1
			}
		}
	}

	// Perform tree-based Rotation.
	diffRotsMap := make(map[int]rlwe.CiphertextQP)
	DiffRots_arr := make([]int, 0)
	unistep := (1 << 31)
	for rot := range totalDiffRots_tmp {
		DiffRots_arr = append(DiffRots_arr, rot)
		if (rot*sign < unistep) && rot != 0 {
			unistep = rot * sign
		}
	}
	if sign < 0 {
		sort.Slice(DiffRots_arr, func(i, j int) bool {
			return DiffRots_arr[i] > DiffRots_arr[j]
		})
	} else {
		sort.Slice(DiffRots_arr, func(i, j int) bool {
			return DiffRots_arr[i] < DiffRots_arr[j]
		})
	}

	levelQ := utils.MinInt(ctIn.Level(), collapsed.level)

	// now := time.Now()
	eval.TreebasedHoistedRotation(ctIn, levelQ, unistep*sign, DiffRots_arr, 0, diffRotsMap, sk)
	// fmt.Printf("TreebasedHoistedrotation uses %s\n", time.Since(now))
	// Now we get all the rotation we need in the CiphertextQP Form.
	// It's time to map them to target nodes in nexcollpase using masks.
	ringQ := eval.params.RingQ()
	ringP := eval.params.RingP()
	ringQP := eval.params.RingQP()
	levelP := len(ringP.Modulus) - 1

	QiOverF := eval.params.QiOverflowMargin(levelQ) >> 1
	PiOverF := eval.params.PiOverflowMargin(levelP) >> 1

	cnt := 0
	Numrescale := 0
	NumMul := 0
	for i_g := range nexcollapse {
		// entering target level of the i_g-th group
		for t := range nexcollapse[i_g] {
			// entering target type of node
			for idx := range nexcollapse[i_g][t] {

				// locate one specific node.

				// the 0-th node of standby type at the first group should only have one map for Rot(0) full with 1.
				if i_g == 0 && t == 1 && idx == 0 {
					nex[i_g][t][idx] = NewCiphertext(eval.params, 1, levelQ-1) // its level should align with thoes need to do rescale.
					nex[i_g][t][idx].Copy(ctIn)
					continue
				} else if len(nexcollapse[i_g][t][idx]) == 0 {
					continue
				}

				cnt = 0
				// first ModUp this node for rotation step 0.
				ctInTmp0 := ctIn.Value[0].CopyNew()
				ctInTmp1 := ctIn.Value[1].CopyNew()
				ringQ.MulScalarBigintLvl(levelQ, ctInTmp0, ringP.ModulusAtLevel[levelP], ctInTmp0)
				ringQ.MulScalarBigintLvl(levelQ, ctInTmp1, ringP.ModulusAtLevel[levelP], ctInTmp1)
				// Accumulation in QP
				tmp0QP := eval.BuffQP[1].CopyNew()
				tmp1QP := eval.BuffQP[1].CopyNew()
				ctQP := rlwe.CiphertextQP{Value: [2]ringqp.Poly{{Q: tmp0QP.Q, P: tmp0QP.P}, {Q: tmp1QP.Q, P: tmp1QP.P}}}
				for m_id, mask := range nexcollapse[i_g][t][idx] {

					// locate one mask in the node.
					NumMul++

					if m_id == 0 {
						if cnt == 0 {
							ringQ.MulCoeffsMontgomeryConstantLvl(levelQ, mask.Q, ctInTmp0, tmp0QP.Q)
							ringQ.MulCoeffsMontgomeryConstantLvl(levelQ, mask.Q, ctInTmp1, tmp1QP.Q)
							tmp0QP.P.Zero()
							tmp1QP.P.Zero()
						} else {
							// if i_g == 0 && t == 1 && idx == 0 {
							// panic("the 0-th node of type _cc at the first group should only have one map for Rot(0) full with 1.")
							// }
							ringQ.MulCoeffsMontgomeryAndAddNoModLvl(levelQ, mask.Q, ctInTmp0, tmp0QP.Q)
							ringQ.MulCoeffsMontgomeryAndAddNoModLvl(levelQ, mask.Q, ctInTmp1, tmp1QP.Q)
						}
					} else {
						// if i_g == 0 && t == 1 && idx == 0 {
						// panic("the 0-th node of type _cc at the first group should only have one map for Rot(0) full with 1.")
						// }
						if cnt == 0 {
							ringQP.MulCoeffsMontgomeryConstantLvl(levelQ, levelP, mask, diffRotsMap[m_id].Value[0], tmp0QP)
							ringQP.MulCoeffsMontgomeryConstantLvl(levelQ, levelP, mask, diffRotsMap[m_id].Value[1], tmp1QP)
						} else {
							ringQP.MulCoeffsMontgomeryConstantAndAddNoModLvl(levelQ, levelP, mask, diffRotsMap[m_id].Value[0], tmp0QP)
							ringQP.MulCoeffsMontgomeryConstantAndAddNoModLvl(levelQ, levelP, mask, diffRotsMap[m_id].Value[1], tmp1QP)
						}
					}

					if cnt%QiOverF == QiOverF-1 {
						ringQ.ReduceLvl(levelQ, tmp0QP.Q, tmp0QP.Q)
						ringQ.ReduceLvl(levelQ, tmp1QP.Q, tmp1QP.Q)
					}
					if cnt%PiOverF == PiOverF-1 {
						ringP.ReduceLvl(levelP, tmp0QP.P, tmp0QP.P)
						ringP.ReduceLvl(levelP, tmp1QP.P, tmp1QP.P)
					}

					cnt++
				}

				if cnt%QiOverF != 0 {
					ringQ.ReduceLvl(levelQ, tmp0QP.Q, tmp0QP.Q)
					ringQ.ReduceLvl(levelQ, tmp1QP.Q, tmp1QP.Q)
				}
				if cnt%PiOverF != 0 {
					ringP.ReduceLvl(levelP, tmp0QP.P, tmp0QP.P)
					ringP.ReduceLvl(levelP, tmp1QP.P, tmp1QP.P)
				}

				// Finish all aggregation, will do a ModDown combined with Rescale here.
				//if !(i_g == 0 && t == 1 && idx == 0) {
				eval.ModDownAndRescale(levelQ, ctQP, nex[i_g][t][idx])
				Numrescale++
				//}

				// only for debug ------------------------------
				/*
					ct := nex[i_g][t][idx]
					fmt.Printf("Aggregated nex[%d][%d][%d] with Scale %d, Modlv %d\n", i_g, t, idx, int(math.Log2(ct.Scale.Float64())), ct.Level())
					decryptor := NewDecryptor(eval.params, sk)
					encoder := NewEncoder(eval.params)
					pt := decryptor.DecryptNew(ct)
					values := encoder.Decode(pt, eval.params.LogSlots())
					realpart := make([]float64, len(values))
					for i := range values {
						realpart[i] = real(values[i])
					}
					fmt.Printf("[")
					for i := 0; i < len(realpart); i++ {
						fmt.Printf("%.0f ", realpart[i])
					}
					fmt.Printf("]\n")
					fmt.Printf("\n")
				*/
				// only for debug ------------------------------

			}
		}
	}
	fmt.Printf("%d MergedRescale&ModDown, %d QPMul \n", Numrescale, NumMul)
	// runtime.GC()
}

func (eval *evaluator) TreebasedHoistedRotation(ctIn *rlwe.Ciphertext, level int, unitstep int, DiffRots []int, rotated int, resultmap map[int]rlwe.CiphertextQP, sk *rlwe.SecretKey) {

	/*
		if lower == 0 {
			resultmap[rotated] = ctIn
		}
	*/

	levelQ := level                    //  ctIn.Level()             // ctIn.Value[0].LevelQ() // this should be equal to ctIn.Value[1].LevelQ()
	levelP := eval.params.PCount() - 1 // ctIn.Value[0].LevelP() // this should be equal to ctIn.Value[1].LevelP()
	// leftmost := lower
	// ringQP := eval.params.RingQP()

	// Do rotation here:
	// var ctNext rlwe.CiphertextQP
	// c0NextQP := ringQP.NewPolyLvl(levelQ, levelP)
	// c1NextQP := ringQP.NewPolyLvl(levelQ, levelP)
	// ctNext = rlwe.CiphertextQP{Value: [2]ringqp.Poly{c0NextQP, c1NextQP}}
	// First, finish the ModDown of ctIn:
	// eval.BasisExtender.ModDownQPtoQNTT(levelQ, levelP, ctIn.Value[0].Q, ctIn.Value[0].P, ctNext.Value[0].Q)
	// eval.BasisExtender.ModDownQPtoQNTT(levelQ, levelP, ctIn.Value[1].Q, ctIn.Value[1].P, ctNext.Value[1].Q)

	ctNext := NewCiphertext(eval.params, 1, level)
	ctNext.MetaData = ctIn.MetaData
	ctNext.Scale = ctIn.Scale

	/*
		new_DiffRots := make([]int, len(DiffRots))
		for i := range new_DiffRots {
			new_DiffRots[i] = DiffRots[i] - leftmost
		}
	*/
	// Split new_numbers into smaller groups
	new_lower := 0
	new_upper := unitstep // 1
	subgroups := make([][]int, 0)
	new_lowers := make([]int, 0)
	new_uppers := make([]int, 0)
	if unitstep < 0 {
		for new_lower >= DiffRots[len(DiffRots)-1] {
			subgroup := make([]int, 0)
			for i := range DiffRots {
				if (new_lower >= DiffRots[i]) && (DiffRots[i] > new_upper) {
					subgroup = append(subgroup, DiffRots[i])
				}
			}
			subgroups = append(subgroups, subgroup)
			new_lowers = append(new_lowers, new_lower)
			new_uppers = append(new_uppers, new_upper)
			// update for next subgroup.
			new_lower = new_upper
			new_upper = new_upper * 2
		}
	} else {
		for new_lower <= DiffRots[len(DiffRots)-1] {
			subgroup := make([]int, 0)
			for i := range DiffRots {
				if (new_lower <= DiffRots[i]) && (DiffRots[i] < new_upper) {
					subgroup = append(subgroup, DiffRots[i])
				}

			}
			/*
				for i := range new_DiffRots {
					if (new_lower <= new_DiffRots[i]) && (new_DiffRots[i] < new_upper) {
						subgroup = append(subgroup, new_DiffRots[i])
					}
				}
			*/
			subgroups = append(subgroups, subgroup)
			new_lowers = append(new_lowers, new_lower)
			new_uppers = append(new_uppers, new_upper)
			// update for next subgroup.
			new_lower = new_upper
			new_upper = new_upper * 2
		}
	}

	if !(len(subgroups) == 1 && len(subgroups[0]) == 1 && subgroups[0][0] == 0) {

		// Perform parallel Hoisted Rotation no ModDown on diffRots:
		eval.DecomposeNTT(levelQ, levelP, eval.params.PCount(), ctIn.Value[1], ctIn.IsNTT, eval.BuffDecompQP)
		ctNextMap := eval.RotateHoistedNoModDownNew(levelQ, new_lowers, ctIn.Value[0], eval.BuffDecompQP)
		// fmt.Printf("DecomposeNTT+RotatedHoistedNoModDownNew\n")

		for i := 0; i < len(subgroups); i++ {
			onlyzero := true
			for j := range subgroups[i] {
				subgroups[i][j] = subgroups[i][j] - new_lowers[i]
				if subgroups[i][j] != 0 {
					onlyzero = false
				}
			}
			if _, exists := ctNextMap[new_lowers[i]]; exists {
				// Load the NoMOdDownVer to resultmap:
				resultmap[rotated+new_lowers[i]] = ctNextMap[new_lowers[i]]

				// check if we really need to dive down for this subgroup.
				// If this subgroup only has zero in it, then no need for diving down.
				if onlyzero {
					continue
				}

				// Complete ModDown for next level of Rots.
				ctTmp := ctNextMap[new_lowers[i]]

				// now := time.Now()
				eval.BasisExtender.ModDownQPtoQNTT(levelQ, levelP, ctTmp.Value[0].Q, ctTmp.Value[0].P, ctNext.Value[0])
				eval.BasisExtender.ModDownQPtoQNTT(levelQ, levelP, ctTmp.Value[1].Q, ctTmp.Value[1].P, ctNext.Value[1])
				// fmt.Printf("ModDownQPtoQNTT %s\n", time.Since(now))

				// only for debug ------------------------------
				/*
					ct := ctNext
					fmt.Printf("Pre-rotated %d steps ct with Scale %d, Modlv %d\n", rotated+new_lowers[i], int(math.Log2(ct.Scale.Float64())), ct.Level())
					decryptor := NewDecryptor(eval.params, sk)
					encoder := NewEncoder(eval.params)
					pt := decryptor.DecryptNew(ct)
					values := encoder.Decode(pt, eval.params.LogSlots())
					realpart := make([]float64, len(values))
					for i := range values {
						realpart[i] = real(values[i])
					}
					fmt.Printf("[")
					for i := 0; i < len(realpart); i++ {
						fmt.Printf("%.0f ", realpart[i])
					}
					fmt.Printf("]\n")
					fmt.Printf("\n")
				*/
				// only for debug ------------------------------

				eval.TreebasedHoistedRotation(ctNext, level, unitstep, subgroups[i], rotated+new_lowers[i], resultmap, sk)
			}
		}
		/*
			eval.DecomposeNTT(levelQ, levelP, eval.params.PCount(), ctNext.Value[1].Q, ctNext.IsNTT, eval.BuffDecompQP)
			ctNextMap := eval.RotateHoistedNoModDownNew(levelQ, new_lowers[1:], ctNext.Value[0].Q, eval.BuffDecompQP)
			for i := 1; i < len(subgroups); i++ {
				for j := range subgroups[i] {
					subgroups[i][j] = subgroups[i][j] - new_lowers[i]
				}
				// Second, Perform parallel Hoisted Rotation no ModDown on diffRots:
				eval.TreebasedHoistedRotation(ctNextMap[new_lowers[i]], subgroups[i], n, new_lowers[i], new_uppers[i], rotated+leftmost, resultmap, sk)
			}
		*/
	}

}

func GenCollpasedLevelsInfoBottom(encoder Encoder, nexcollapse_Pt [][]map[int]map[int][]int, collapse_depth int, level int, scale rlwe.Scale, logslots int) (rslt collapsedlevelsInfoB) {
	enc, ok := encoder.(*encoderComplex128)
	if !ok {
		panic("cannot GenLinearTransform: encoder should be an encoderComplex128")
	}
	params := enc.params
	rslt = collapsedlevelsInfoB{nexcollapse: nil, nexcollapse_pt: nexcollapse_Pt, totalDiffRots: nil, level: level, collapse_depth: collapse_depth, scale: scale}
	rslt.totalDiffRots = make(map[int]int)
	rslt.nexcollapse = make([][]map[int]map[int]*ring.Poly, len(nexcollapse_Pt))
	levelQ := level
	// levelP := params.PCount() - 1
	for i_g := range nexcollapse_Pt {
		// entering target level of the i_g-th group
		rslt.nexcollapse[i_g] = make([]map[int]map[int]*ring.Poly, len(nexcollapse_Pt[i_g]))
		for t := range nexcollapse_Pt[i_g] {
			// entering target type of node
			rslt.nexcollapse[i_g][t] = make(map[int]map[int]*ring.Poly)
			for idx := range nexcollapse_Pt[i_g][t] {
				// entering a specific node:
				rslt.nexcollapse[i_g][t][idx] = make(map[int]*ring.Poly)
				for m_id := range nexcollapse_Pt[i_g][t][idx] {

					// entering a mask in the node, the mask corresponding to one ciphertext before rotation.
					// the m_id is the rotation step of the ciphertext after masking and aggregation.
					mask_float64 := make([]float64, len(nexcollapse_Pt[i_g][t][idx][m_id]))
					for i := range mask_float64 {
						mask_float64[i] = float64(nexcollapse_Pt[i_g][t][idx][m_id][i])
					}
					rslt.nexcollapse[i_g][t][idx][m_id] = params.RingQ().NewPolyLvl(levelQ)
					enc.Embed(mask_float64, logslots, scale, true, rslt.nexcollapse[i_g][t][idx][m_id])

					// record the rotation step.
					rslt.totalDiffRots[m_id] = 1
				}
			}
		}
	}
	return
}

func (eval *evaluator) MultiGroupNetworkCollapseBottomNew(collapsed collapsedlevelsInfoB, sign int, n int, sk *rlwe.SecretKey, nex [][]map[int]*rlwe.Ciphertext) (ctOut *rlwe.Ciphertext) {
	// perform Upper levels' Collpasing
	// first is to prepare all rotation steps.
	// Restricted to only use rotation keys corresponding to power of 2.
	// The logic to judge whether we need to rotate a power of 2 goes like this:
	// 1. Mod the rots to let them have equal sign.
	totalDiffRots := collapsed.totalDiffRots
	nexcollapse := collapsed.nexcollapse
	totalDiffRots_tmp := make(map[int]int)
	if sign == -1 {
		for dist := range totalDiffRots {
			if dist >= n || dist <= -n {
				panic("|invalid dist| >= n")
			} else if dist > 0 {
				totalDiffRots_tmp[dist-n] = 1
			} else {
				totalDiffRots_tmp[dist] = 1
			}
		}
	} else {
		for dist := range totalDiffRots {
			if dist >= n || dist <= -n {
				panic("|invalid dist| >= n")
			} else if dist < 0 {
				totalDiffRots_tmp[dist+n] = 1
			} else {
				totalDiffRots_tmp[dist] = 1
			}
		}
	}

	levelQ := collapsed.level // utils.MinInt(ctOut.Level(), collapsed.level)

	// Now we get all the rotation we need in the CiphertextQP Form.
	// It's time to map them to target nodes in nexcollpase using masks.
	ringQ := eval.params.RingQ()
	// ringP := eval.params.RingP()
	// ringQP := eval.params.RingQP()
	// levelP := len(ringP.Modulus) - 1

	QiOverF := eval.params.QiOverflowMargin(levelQ) >> 1
	// PiOverF := eval.params.PiOverflowMargin(levelP) >> 1

	// create buffer for rotation.
	ctBuff := make(map[int]*rlwe.Ciphertext, len(totalDiffRots_tmp))
	cntBuff := make(map[int]int, len(totalDiffRots_tmp))
	for i := range totalDiffRots_tmp {
		ctBuff[i] = NewCiphertext(eval.params, 1, levelQ)
		ctBuff[i].MetaData = nex[0][0][0].MetaData
		ctBuff[i].Scale = collapsed.scale.Mul(nex[0][0][0].Scale)
	}

	// cnt := 0
	// Numrescale := 0
	now_botMul := time.Now()
	NumMul := 0
	for i_g := range nexcollapse {
		// entering target level of the i_g-th group
		for t := range nexcollapse[i_g] {
			// entering target type of node
			for idx := range nexcollapse[i_g][t] {

				// locate one specific node.

				// the 0-th node of standby type at the first group should only have one map for Rot(0) full with 1.
				if len(nexcollapse[i_g][t][idx]) == 0 {
					continue
				}

				/*
					if nex[i_g][t][idx].Level() > 15 {
						for m_id, mask := range collapsed.nexcollapse_pt[i_g][t][idx] {
							_ = m_id
							for e_id, e := range mask {
								_ = e_id
								if e != 0 {
									panic("Got unexpected mask here.")
								}
							}
						}
					}
				*/

				// Accumulation in Q
				for m_id, mask := range nexcollapse[i_g][t][idx] {

					// locate one mask in the node.
					NumMul++
					if cntBuff[m_id] == 0 {
						ringQ.MulCoeffsMontgomeryConstantLvl(levelQ, mask, nex[i_g][t][idx].Value[0], ctBuff[m_id].Value[0])
						ringQ.MulCoeffsMontgomeryConstantLvl(levelQ, mask, nex[i_g][t][idx].Value[1], ctBuff[m_id].Value[1])
					} else {
						ringQ.MulCoeffsMontgomeryAndAddNoModLvl(levelQ, mask, nex[i_g][t][idx].Value[0], ctBuff[m_id].Value[0])
						ringQ.MulCoeffsMontgomeryAndAddNoModLvl(levelQ, mask, nex[i_g][t][idx].Value[1], ctBuff[m_id].Value[1])
					}

					if cntBuff[m_id]%QiOverF == QiOverF-1 {
						ringQ.ReduceLvl(levelQ, ctBuff[m_id].Value[0], ctBuff[m_id].Value[0])
						ringQ.ReduceLvl(levelQ, ctBuff[m_id].Value[1], ctBuff[m_id].Value[1])
					}
					cntBuff[m_id]++

					// only for debug ------------------------------
					/*
						if cntBuff[m_id]%QiOverF != 0 {
							ringQ.ReduceLvl(levelQ, ctBuff[m_id].Value[0], ctBuff[m_id].Value[0])
							ringQ.ReduceLvl(levelQ, ctBuff[m_id].Value[1], ctBuff[m_id].Value[1])
						}
						ct := ctBuff[m_id] // nex[i_g][t][idx]
						fmt.Printf("ctBuff[%d] Aggregates nex[%d][%d][%d] with Scale %d, Modlv %d \n", m_id, i_g, t, idx, int(math.Log2(ct.Scale.Float64())), ct.Level())
						decryptor := NewDecryptor(eval.params, sk)
						encoder := NewEncoder(eval.params)
						pt := decryptor.DecryptNew(ct)
						values := encoder.Decode(pt, eval.params.LogSlots())
						realpart := make([]float64, len(values))
						for i := range values {
							realpart[i] = real(values[i])
						}
						fmt.Printf("[")
						for i := 0; i < len(realpart); i++ {
							fmt.Printf("%.0f ", realpart[i])
						}
						fmt.Printf("]\n")
						fmt.Printf("\n")
					*/
					// only for debug ------------------------------
				}

			}
		}
	}

	levelQ_new := levelQ
	var Metadata rlwe.MetaData
	cnt_assign := 0
	for i := range ctBuff {
		if cntBuff[i]%QiOverF != 0 {
			ringQ.ReduceLvl(levelQ, ctBuff[i].Value[0], ctBuff[i].Value[0])
			ringQ.ReduceLvl(levelQ, ctBuff[i].Value[1], ctBuff[i].Value[1])
		}
		// ctBuff[i].Scale = collapsed.scale.Mul(nex[0][0][0].Scale)
		eval.Rescale(ctBuff[i], eval.params.DefaultScale(), ctBuff[i])
		if levelQ_new > ctBuff[i].Level() {
			levelQ_new = ctBuff[i].Level()
		}
		if cnt_assign == 0 {
			Metadata = ctBuff[i].MetaData
			cnt_assign++
		}
	}
	fmt.Printf("BotMult+Rescale: %s \n", time.Since(now_botMul))

	ringP := eval.params.RingP()
	ringQP := eval.params.RingQP()
	levelP := eval.params.PCount() - 1
	nbPi := eval.params.PCount()
	QiOverF_new := eval.params.QiOverflowMargin(levelQ_new) >> 1
	PiOverF := eval.params.PiOverflowMargin(levelP) >> 1

	polyOutQP0 := ringQP.NewPolyLvl(levelQ_new, levelP)
	polyOutQP1 := ringQP.NewPolyLvl(levelQ_new, levelP)
	ctOutQP := rlwe.CiphertextQP{Value: [2]ringqp.Poly{{Q: polyOutQP0.Q, P: polyOutQP0.P}, {Q: polyOutQP1.Q, P: polyOutQP1.P}}}

	now_botRot := time.Now()
	cnt_sum := 0
	for i := range ctBuff {
		rot_rest := i
		var tmpQP rlwe.CiphertextQP
		for step := 1 * sign; rot_rest != 0; step *= 2 {
			binary_1 := -((-step) & (-rot_rest))
			if step != (binary_1) {
				continue
			}
			if rot_rest-step == 0 {
				eval.DecomposeNTT(ctBuff[i].Level(), levelP, nbPi, ctBuff[i].Value[1], ctBuff[i].IsNTT, eval.BuffDecompQP)
				tmpQP = eval.RotateHoistedNoModDownNew(ctBuff[i].Level(), []int{step}, ctBuff[i].Value[0], eval.BuffDecompQP)[step]
				if cnt_sum > 0 {
					ringQP.AddNoModLvl(levelQ_new, levelP, tmpQP.Value[0], ctOutQP.Value[0], ctOutQP.Value[0])
					ringQP.AddNoModLvl(levelQ_new, levelP, tmpQP.Value[1], ctOutQP.Value[1], ctOutQP.Value[1])
				} else {
					ringQP.CopyLvl(levelQ_new, levelP, tmpQP.Value[0], ctOutQP.Value[0])
					ringQP.CopyLvl(levelQ_new, levelP, tmpQP.Value[1], ctOutQP.Value[1])
				}
				if cnt_sum%QiOverF_new == QiOverF_new-1 {
					ringQ.ReduceLvl(levelQ_new, ctOutQP.Value[0].Q, ctOutQP.Value[0].Q)
					ringQ.ReduceLvl(levelQ_new, ctOutQP.Value[1].Q, ctOutQP.Value[1].Q)
					// ringQ.ReduceLvl(levelQ)
				}
				if cnt_sum%PiOverF == PiOverF-1 {
					ringP.ReduceLvl(levelP, ctOutQP.Value[0].P, ctOutQP.Value[0].P)
					ringP.ReduceLvl(levelP, ctOutQP.Value[1].P, ctOutQP.Value[1].P)
					// ringQ.ReduceLvl(levelQ)
				}
				cnt_sum++
				// NoModDown Rot
			} else {
				// ModDon Rot
				eval.Rotate(ctBuff[i], step, ctBuff[i])
			}
			rot_rest = rot_rest - step
		}
	}
	if cnt_sum%QiOverF_new != 0 {
		ringQ.ReduceLvl(levelQ_new, ctOutQP.Value[0].Q, ctOutQP.Value[0].Q)
		ringQ.ReduceLvl(levelQ_new, ctOutQP.Value[1].Q, ctOutQP.Value[1].Q)
	}
	if cnt_sum%PiOverF != 0 {
		ringP.ReduceLvl(levelP, ctOutQP.Value[0].P, ctOutQP.Value[0].P)
		ringP.ReduceLvl(levelP, ctOutQP.Value[1].P, ctOutQP.Value[1].P)
	}
	ctOut = NewCiphertext(eval.params, 1, levelQ_new)

	eval.BasisExtender.ModDownQPtoQNTT(levelQ_new, levelP, ctOutQP.Value[0].Q, ctOutQP.Value[0].P, ctOut.Value[0])
	eval.BasisExtender.ModDownQPtoQNTT(levelQ_new, levelP, ctOutQP.Value[1].Q, ctOutQP.Value[1].P, ctOut.Value[1])

	if _, exists := ctBuff[0]; exists {
		eval.Add(ctBuff[0], ctOut, ctOut)
	}

	ctOut.MetaData = Metadata
	fmt.Printf("BotRot: %s\n", time.Since(now_botRot))
	return
	// runtime.GC()
}

func (eval *evaluator) MultiGroupNetworkTopDownNewV4LvlCollapse(ctIn *rlwe.Ciphertext, RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, lv_begin, lv_end, inc int, pre, nex [][]map[int]*rlwe.Ciphertext, nex_cnt [][]map[int]int, UpCInfo collapsedlevelsInfo, BotCInfo collapsedlevelsInfoB, ecd_levelEachGroup []map[int]int, sk *rlwe.SecretKey) (ctOut *rlwe.Ciphertext) {
	// memory pre-allocate version of V2.
	var _a = 0
	// var _c = 1

	var _aa = 0
	var _ac = 1
	var _ca = 2
	var _cc = 3

	// Initialize it with input ciphertext.
	// pre[0][_a][0].Copy(ctIn)
	// pre[0][_c][0] = ctIn

	// Get ring to perform lazy Mult and Add
	ringQ := eval.params.RingQ()
	Maxlevel := ctIn.Level()
	_ = Maxlevel

	// Temp Buffers:
	var ctTmp1 *rlwe.Ciphertext
	var ctTmp2 *rlwe.Ciphertext
	var _x, _y, s, t int
	var ecd_level, QiOverFMul, cnt int
	var lv_end_rlt int
	var visitOrder = [4]int{_ac, _aa, _ca, _cc}
	var minScale, divScale rlwe.Scale
	var lv int

	now_total := time.Now()
	if UpCInfo.collapse_depth > 0 {
		now_upper := time.Now()
		eval.MultiGroupNetworkCollapseUpper(ctIn, UpCInfo, -1, 1<<(eval.params.logSlots), sk, pre)
		// pre = nex
		fmt.Printf("UpperCollpased using %s\n", time.Since(now_upper))
		lv = lv_begin + UpCInfo.collapse_depth + 1
	} else {
		pre[0][0][0].Resize(1, ecd_levelEachGroup[0][lv_begin])
		pre[0][0][0].Copy(ctIn)

		pre[0][1][0].Resize(1, ecd_levelEachGroup[0][lv_begin])
		pre[0][1][0].Copy(ctIn)
		lv = lv_begin
	}

	for ; lv < lv_end-BotCInfo.collapse_depth; lv += inc { // lv := lv_begin + UpCInfo.collapse_depth + 1; lv < lv_end-BotCInfo.collapse_depth; lv += inc {
		now := time.Now()
		// Rotation(Combined with Rescale):
		for m := range RRot {

			for k := range pre[m][_a] {
				if step, exists := RRot[m][lv]; exists {
					ctTmp1 = pre[m][_a][k]

					minScale = eval.params.DefaultScale().Div(rlwe.NewScale(2))
					divScale = pre[m][_a][k].Scale.Div(rlwe.NewScale(eval.params.RingQ().Modulus[pre[m][_a][k].Level()]))

					if lv == lv_begin || divScale.Cmp(minScale) == -1 {
						eval.Rotate(ctTmp1, step, ctTmp1)
						fmt.Printf("Rotation performed on Level %d\n", ctTmp1.Level())
					} else {
						if step == 0 {
							eval.Rescale(ctTmp1, eval.params.DefaultScale(), ctTmp1)
						}
						fmt.Printf("Rotation+Rescale to perform on Level %d\n", ctTmp1.Level())
						eval.RotateHoistedAndRescaleOne(ctTmp1, step, ctTmp1)
						// fmt.Printf("Rotation performed on Level %d\n", ctTmp1.Level())
					}
				}
			}
		}

		if lv == lv_end-BotCInfo.collapse_depth-1 {
			elapsed_tmp := time.Since(now)
			fmt.Printf("lv %d consume %s \n", lv, elapsed_tmp)
			// nex = pre
			break // no need to do redundant masking between rotation and bottomcollapsing.
		}

		// Masking to next level:
		for i_g := len(RRot) - 1; i_g >= 0; i_g-- {

			// 0903:
			lv_end_rlt = len(MMaskpt[i_g][i_g]) - 1

			if lv < lv_end_rlt {
				// Get encoding level of this round (after rotation's rescale)
				ecd_level = ecd_levelEachGroup[i_g][lv] // ecd_level = Maxlevel - lv + lv_begin // ecd_level = Maxlevel - lv + lv_begin - (lv_end - lv_end_rlt)
				// Get Maximum NUmber of Addition before Overflow:
				QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
			}

			for m := i_g; m >= 0; m-- {
				if len(MMaskpt[m][i_g]) <= lv || MMaskpt[m][i_g][lv] == nil {
					continue
				}
				// handle mapping:
				for _, cls := range visitOrder {

					for k, v := range MMaskpt[m][i_g][lv][cls] {
						_x = cls / 2
						_y = cls % 2
						fmt.Sscanf(k, "%d:%d", &s, &t)

						ctTmp1 = pre[m][_x][s]

						switch {
						case (cls == _aa || cls == _ca):
							/*
								if nex[i_g][_y][t] == nil {
									nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ctTmp1.Level())
								}
							*/
							ctTmp2 = nex[i_g][_y][t]
							if nex_cnt[i_g][_y][t] == 0 {
								// nex_cnt[i_g][_y][t] = 0
								ctTmp2.MetaData = ctTmp1.MetaData
								ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
								ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							} else {
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
								ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							}
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac && lv == lv_end_rlt-1) || (cls == _cc && lv == lv_end_rlt-1):
							/*
								if nex[i_g][_y][t] == nil {
									nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ecd_level)
								}
							*/
							ctTmp2 = nex[i_g][_y][t]
							ctTmp2.MetaData = ctTmp1.MetaData
							ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
							ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac):
							if lv < lv_end_rlt-1 {
								nex[i_g][_y][t].Copy(ctTmp1) // = ctTmp1.CopyNew()
							} else if lv == lv_end_rlt {
								nex[i_g][_x][t] = ctTmp1 // we only do this once, so no worry for modification.
							}

						case (cls == _cc):
							if lv < lv_end_rlt-1 {
								nex[i_g][_y][t] = ctTmp1 // we only do this once per ciphertext, so no worry for modification.
							} else if lv == lv_end_rlt {
								if cnt == 0 {
									// nex_cnt[i_g][_y][t] = 0
									ctOut = ctTmp1.CopyNew()
								} else {
									ctTmp2 = ctOut
									// debug
									/*
										if ctTmp1.Level() != ctTmp2.Level() {
											panic("???")
										}
									*/
									ringQ.AddNoMod(ctTmp1.Value[0], ctTmp2.Value[0], ctTmp2.Value[0])
									ringQ.AddNoMod(ctTmp1.Value[1], ctTmp2.Value[1], ctTmp2.Value[1])
								}
								if cnt%QiOverFMul == QiOverFMul-1 {
									ctTmp2 = ctOut
									ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
									ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
								}
								cnt++
							}
						}
					}

				}
			}

			if lv < lv_end_rlt {
				m := i_g
				for k := range nex_cnt[m] {
					for t, cnt_tmp := range nex_cnt[m][k] {
						if cnt_tmp%QiOverFMul != 0 {
							ctTmp2 = nex[m][k][t]
							ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
							ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
						}
						delete(nex_cnt[m][k], t)
					}
				}
				pre[m] = nex[m]
			}
		}
		elapsed_tmp := time.Since(now)
		fmt.Printf("lv %d consume %s \n", lv, elapsed_tmp)
	}

	// before entering bottom collpased levels, we need to do one rescale.

	now_bot := time.Now()
	ctOut = eval.MultiGroupNetworkCollapseBottomNew(BotCInfo, -1, 1<<eval.params.logSlots, sk, pre)
	fmt.Printf("CollpasedBottom using %s\n", time.Since(now_bot))
	fmt.Printf("Total network using %s\n", time.Since(now_total))
	return
}

// backup:
/*
	// Rotation(Combined with Rescale):
	for m := range RRot {
		for k := range pre[m][_a] {
			if step, exists := RRot[m][lv]; exists {
				ctTmp1 = pre[m][_a][k]

				minScale = eval.params.DefaultScale().Div(rlwe.NewScale(2))
				divScale = pre[m][_a][k].Scale.Div(rlwe.NewScale(eval.params.RingQ().Modulus[pre[m][_a][k].Level()]))

				if lv == lv_begin || divScale.Cmp(minScale) == -1 {
					eval.Rotate(ctTmp1, step, ctTmp1)
				} else {
					if step == 0 {
						eval.Rescale(ctTmp1, eval.params.DefaultScale(), ctTmp1)
					}
					eval.RotateHoistedAndRescaleOne(ctTmp1, step, ctTmp1)
				}
			}
		}
		// Mask on ciphertext:
		for i_g := range MMaskpt[m] {
			// skip empty ones
			if len(MMaskpt[m][i_g]) <= lv || MMaskpt[m][i_g][lv] == nil {
				continue
			}
			// handle mapping:
			for _, cls := range visitOrder {

				for k, v := range MMaskpt[m][i_g][lv][cls] {
					_x = cls / 2
					_y = cls % 2
					fmt.Sscanf(k, "%d:%d", &s, &t)

					ctTmp1 = pre[m][_x][s]

					switch {
					case (cls == _aa || cls == _ca):
						if nex[i_g][_y][t] == nil {
							nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ctTmp1.Level())
						}
						ctTmp2 = nex[i_g][_y][t]
						if nex_cnt[i_g][_y][t] == 0 {
							// nex_cnt[i_g][_y][t] = 0
							ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
							ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
							ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
						} else {
							ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
							ringQ.MulCoeffsMontgomeryConstantAndAddNoModLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
						}
						if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
							ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
							ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
						}
						nex_cnt[i_g][_y][t]++

					case (cls == _ac && lv == lv_end-1) || (cls == _cc && lv == lv_end-1):
						if nex[i_g][_y][t] == nil {
							nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ecd_level)
						}
						ctTmp2 = nex[i_g][_y][t]
						ctTmp2.Scale = ctTmp1.Scale.Mul(v.Scale)
						ctTmp2.Resize(ctTmp2.Degree(), ecd_level)
						ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[0], v.Value, ctTmp2.Value[0])
						ringQ.MulCoeffsMontgomeryConstantLvl(ecd_level, ctTmp1.Value[1], v.Value, ctTmp2.Value[1])
						if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
							ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
							ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
						}
						nex_cnt[i_g][_y][t]++

					case (cls == _ac):
						if lv < lv_end-1 {
							nex[i_g][_y][t] = ctTmp1.CopyNew()
						} else if lv == lv_end {
							nex[i_g][_x][t] = ctTmp1 // we only do this once, so no worry for modification.
						}

					case (cls == _cc):
						if lv < lv_end-1 {
							nex[i_g][_y][t] = ctTmp1 // we only do this once per ciphertext, so no worry for modification.
						} else if lv == lv_end {
							if nex_cnt[i_g][_y][t] == 0 {
								// nex_cnt[i_g][_y][t] = 0
								nex[i_g][_y][t] = ctTmp1
							} else {
								ctTmp2 = nex[i_g][_y][t]
								// debug
								if ctTmp1.Level() != ctTmp2.Level() {
									panic("???")
								}
								ringQ.AddNoMod(ctTmp1.Value[0], ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.AddNoMod(ctTmp1.Value[1], ctTmp2.Value[1], ctTmp2.Value[1])
							}
							if nex_cnt[i_g][_y][t]%QiOverFMul == QiOverFMul-1 {
								ctTmp2 = nex[i_g][_y][t]
								ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
								ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
							}
							nex_cnt[i_g][_y][t]++
						}
					}
				}

			}

		}




*/

// 0901 backup:
/*
	case (cls == _cc):
		if lv < lv_end-1 {
			nex[i_g][_y][t] = ctTmp1 // we only do this once per ciphertext, so no worry for modification.
		} else if lv == lv_end {
			if nex_cnt[i_g][_y][0] == 0 {
				// nex_cnt[i_g][_y][t] = 0
				nex[i_g][_y][0] = ctTmp1.CopyNew()
			} else {
				ctTmp2 = nex[i_g][_y][0]
				// debug
				if ctTmp1.Level() != ctTmp2.Level() {
					panic("???")
				}
				ringQ.AddNoMod(ctTmp1.Value[0], ctTmp2.Value[0], ctTmp2.Value[0])
				ringQ.AddNoMod(ctTmp1.Value[1], ctTmp2.Value[1], ctTmp2.Value[1])
			}
			if nex_cnt[i_g][_y][0]%QiOverFMul == QiOverFMul-1 {
				ctTmp2 = nex[i_g][_y][0]
				ringQ.Reduce(ctTmp2.Value[0], ctTmp2.Value[0])
				ringQ.Reduce(ctTmp2.Value[1], ctTmp2.Value[1])
			}
			nex_cnt[i_g][_y][0]++
		}




			// cnt := 0
	ecd_level = Maxlevel - lv_end + lv_begin
	QiOverFMul = eval.params.QiOverflowMargin(ecd_level) >> 1
	for m := range pre {
		ctTmp1 = pre[m][_c][0]
		eval.Rescale(ctTmp1, eval.params.DefaultScale(), ctTmp1)
		// debug
		if ctTmp1.Level() != ecd_level {
			panic("????")
		}

		if m == 0 {
			ctOut = ctTmp1
		} else {
			ringQ.AddNoModLvl(ecd_level, ctOut.Value[0], ctTmp1.Value[0], ctOut.Value[0])
			ringQ.AddNoModLvl(ecd_level, ctOut.Value[1], ctTmp1.Value[1], ctOut.Value[1])
		}
		if cnt%QiOverFMul == QiOverFMul-1 {
			ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
			ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
		}
		cnt++

		ctTmp1 = pre[m][_a][0]
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[0], ctTmp1.Value[0], ctOut.Value[0])
		ringQ.AddNoModLvl(ecd_level, ctOut.Value[1], ctTmp1.Value[1], ctOut.Value[1])
		if cnt%QiOverFMul == QiOverFMul-1 {
			ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
			ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
		}
		cnt++
	}
	if cnt%QiOverFMul != 0 {
		ringQ.Reduce(ctOut.Value[0], ctOut.Value[0])
		ringQ.Reduce(ctOut.Value[1], ctOut.Value[1])
	}

	return
*/
