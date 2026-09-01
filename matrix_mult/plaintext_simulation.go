package matrix_mult

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"sort"
	"strconv"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"project1-fhe_extension_v1.0/auxiliary_io"
)

func Sigma_permute(A [][]float64, d int) (rslt [][]float64) {
	i := 0
	j := 0
	rslt = make([][]float64, d)
	for i = 0; i < d; i++ {
		rslt[i] = make([]float64, d)
	}
	for i = 0; i < d; i++ {
		for j = 0; j < d; j++ {
			rslt[i][j] = A[i][(i+j)%d]
		}
	}
	return
}

func Tao_permute(A [][]float64, d int) (rslt [][]float64) {
	i := 0
	j := 0
	rslt = make([][]float64, d)
	for i = 0; i < d; i++ {
		rslt[i] = make([]float64, d)
	}
	for i = 0; i < d; i++ {
		for j = 0; j < d; j++ {
			rslt[i][j] = A[(i+j)%d][j]
		}
	}
	return
}

func Phi_permute(A [][]float64, d int) (rslt [][]float64) {
	i := 0
	j := 0
	rslt = make([][]float64, d)
	for i = 0; i < d; i++ {
		rslt[i] = make([]float64, d)
	}
	for i = 0; i < d; i++ {
		for j = 0; j < d; j++ {
			rslt[i][j] = A[i][(j+1)%d]
		}
	}
	return
}

func Psi_permute(A [][]float64, d int) (rslt [][]float64) {
	i := 0
	j := 0
	rslt = make([][]float64, d)
	for i = 0; i < d; i++ {
		rslt[i] = make([]float64, d)
	}
	for i = 0; i < d; i++ {
		for j = 0; j < d; j++ {
			rslt[i][j] = A[(i+1)%d][j]
		}
	}
	return
}

func Zero_padding(A [][]float64, row int, col int) (rslt [][]float64) {
	i := 0
	if row == col {
		return A
	}
	if row > col {
		rslt = make([][]float64, row)
		for i = 0; i < row; i++ {
			rslt[i] = make([]float64, row)
			copy(rslt[i], A[i])
		}
		return
	}
	if col > row {
		rslt = make([][]float64, col)
		for i = 0; i < col; i++ {
			rslt[i] = make([]float64, col)
			copy(rslt[i], A[i])
		}
		return
	}
	return nil
}

func Row_ordering(a []float64) (A [][]float64, err error) {
	n := len(a)
	d := int(math.Sqrt(float64(n)))
	if d*d != n {
		return nil, errors.New("err: input vector's length can't be squared into integer")
	}
	A = make([][]float64, d)
	for i := 0; i < d; i++ {
		A[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			A[i][j] = a[i*d+j]
		}
	}
	return
}

func Row_ordering_multiple(a []float64, g int) (A [][]float64, err error) {
	if len(a)%g != 0 {
		return nil, errors.New("err: the length of the input vector is not a multiple of k")
	}
	n := len(a) / g
	d := int(math.Sqrt(float64(n)))
	if n%d != 0 {
		return nil, errors.New("err: input vector's length can't be squared into integer")
	}
	A = make([][]float64, d)
	for i := 0; i < g*d; i++ {
		A[i] = make([]float64, d)
	}
	for k := 0; k < g; k++ {
		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				A[k*d+i][j] = a[g*(d*i+j)+k]
			}
		}
	}
	return
}

func Row_orderingInv(A [][]float64) (a []float64, err error) {
	/*
		if len(A) != len(A[0]) {
			return nil, errors.New("err: input matrix is not a square one")
		}
	*/
	row := len(A)
	col := len(A[0])
	a = make([]float64, row*col)
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			a[i*col+j] = A[i][j]
		}
	}
	return
}

func Row_orderingInvZeroPad(A [][]float64, d int) (a []float64, err error) {
	if d < len(A) || d < len(A[0]) {
		return nil, errors.New("square dimension smaller than original colsize and rowsize")
	}
	row := len(A)
	col := len(A[0])
	a = make([]float64, d*d)
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			a[i*col+j+i*(d-col)] = A[i][j]
		}
	}
	return
}

func Row_orderingInv_multiple(A [][][]float64) (a []float64, err error) {
	g := len(A)
	if len(A[0]) != len(A[0][0]) {
		return nil, errors.New("err: input matrices is not square matrices")
	}
	d := len(A[0])
	a = make([]float64, d*d*g)
	for k := 0; k < g; k++ {
		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				a[g*(d*i+j)+k] = A[k][i][j]
			}
		}
	}
	return
}

func Col_ordering(a []float64) (A [][]float64, err error) {
	n := len(a)
	d := int(math.Sqrt(float64(n)))
	if d*d != n {
		return nil, errors.New("err: input vector's length can't be squared into integer")
	}
	A = make([][]float64, d)
	for i := 0; i < d; i++ {
		A[i] = make([]float64, d)
	}
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			A[j][i] = a[i*d+j]
		}
	}
	return
}

func Col_orderingInv(A [][]float64) (a []float64, err error) {
	if len(A) != len(A[0]) {
		return nil, errors.New("err: input matrix is not a square one")
	}
	d := len(A)
	a = make([]float64, d*d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			a[i*d+j] = A[j][i]
		}
	}
	return
}

func Hadmard_matrix_mult(A [][]float64, B [][]float64) (C [][]float64, err error) {
	if len(A) != len(B) {
		return nil, errors.New("err: A's row number does not match B's row number")
	}
	if len(A[0]) != len(B[0]) {
		return nil, errors.New("err: A's column number does not match B's column number")
	}
	row := len(A)
	col := len(A[0])
	C = make([][]float64, row)
	for i := 0; i < row; i++ {
		C[i] = make([]float64, col)
		for j := 0; j < col; j++ {
			C[i][j] = A[i][j] * B[i][j]
		}
	}
	return
}

func Hadmard_matrix_add(A [][]float64, B [][]float64) (C [][]float64, err error) {
	if len(A) != len(B) {
		return nil, errors.New("err: A's row number does not match B's row number")
	}
	if len(A[0]) != len(B[0]) {
		return nil, errors.New("err: A's column number does not match B's column number")
	}
	row := len(A)
	col := len(A[0])
	C = make([][]float64, row)
	for i := 0; i < row; i++ {
		C[i] = make([]float64, col)
		for j := 0; j < col; j++ {
			C[i][j] = A[i][j] + B[i][j]
		}
	}
	return
}

func SquareMatrix_product_permute_version(A [][]float64, B [][]float64) (C [][]float64, err error) {
	if len(A) != len(B) {
		return nil, errors.New("err: A's row number does not match B's row number")
	}
	if len(A[0]) != len(B[0]) {
		return nil, errors.New("err: A's column number does not match B's column number")
	}
	row := len(A)
	col := len(A[0])
	if row != col {
		return nil, errors.New("err: inpute Matrices is not Square Matrices")
	}
	d := row
	C = make([][]float64, d)
	for i := 0; i < d; i++ {
		C[i] = make([]float64, d)
	}
	for k := 0; k < d; k++ {
		colshiftA := Sigma_permute(A, d)
		rowshiftB := Tao_permute(B, d)
		for i := 0; i < k; i++ {
			colshiftA = Phi_permute(colshiftA, d)
			rowshiftB = Psi_permute(rowshiftB, d)
		}
		var mult_rslt [][]float64
		mult_rslt, err = Hadmard_matrix_mult(colshiftA, rowshiftB)
		if err != nil {
			return nil, err
		}
		C, err = Hadmard_matrix_add(C, mult_rslt)
		if err != nil {
			return nil, err
		}
	}
	return
}

func Gen_sigma_diagonalVecotrs(d int) (U map[int][]float64, err error) {
	if d <= 0 {
		return nil, errors.New("dimension d <= 0 ")
	}
	U = make(map[int][]float64, 2*d-1)
	for k := -d + 1; k < d; k++ {
		U[k] = make([]float64, d*d)
		for i := 0; i < d*d; i++ {
			if k >= 0 {
				if (i-d*k) >= 0 && (i-d*k) < (d-k) {
					U[k][i] = 1
				} else {
					U[k][i] = 0
				}
			} else {
				if (i-(d+k)*d) >= -k && (i-(d+k)*d) < d {
					U[k][i] = 1
				} else {
					U[k][i] = 0
				}
			}
		}
	}
	return
}

func Gen_sigma_diagonalVecotrs_batch(d int, g int) (U map[int][]float64, err error) {
	if d <= 0 || g <= 0 {
		return nil, errors.New("dimension d <= 0 || batchsize g <= 0 ")
	}

	U = make(map[int][]float64, 2*d-1)
	for k := -d + 1; k < d; k++ {
		U[k*g] = make([]float64, d*d*g)
		for i := 0; i < d*d; i++ {
			if k >= 0 {
				if (i-d*k) >= 0 && (i-d*k) < (d-k) {
					for b := 0; b < g; b++ {
						U[k*g][i*g+b] = 1
					}
				} else {
					for b := 0; b < g; b++ {
						U[k*g][i*g+b] = 0
					}
				}
			} else {
				if (i-(d+k)*d) >= -k && (i-(d+k)*d) < d {
					for b := 0; b < g; b++ {
						U[k*g][i*g+b] = 1
					}
				} else {
					for b := 0; b < g; b++ {
						U[k*g][i*g+b] = 0
					}
				}
			}
		}
	}
	return
}

func Gen_tao_diagonalVectors(d int) (U map[int][]float64, err error) {
	if d <= 0 {
		return nil, errors.New("dimension d <= 0 ")
	}
	U = make(map[int][]float64, d)
	for k := 0; k < d; k++ {
		U[d*k] = make([]float64, d*d)
		for i := 0; i < d; i++ {
			U[d*k][k+d*i] = 1
		}
	}
	return
}

func Gen_tao_diagonalVectors_batch(d int, g int) (U map[int][]float64, err error) {
	if d <= 0 || g <= 0 {
		return nil, errors.New("dimension d <= 0 || batchsize g <= 0 ")
	}
	U = make(map[int][]float64, d)
	for k := 0; k < d; k++ {
		U[d*k*g] = make([]float64, d*d*g)
		for i := 0; i < d; i++ {
			for b := 0; b < g; b++ {
				U[d*k*g][(k+d*i)*g+b] = 1
			}
		}
	}
	return
}

func Gen_colShift_diagonalVectors(d int, k int) (U map[int][]float64, err error) {
	if k < 0 || k >= d || d <= 0 { //  k < 1 || k >= d || d <= 0
		return nil, errors.New("dimension d <= 0 or k < 1 or k >= d")
	}
	U = make(map[int][]float64, 2)
	U[k] = make([]float64, d*d)
	U[k-d] = make([]float64, d*d)
	for i := 0; i < d*d; i++ {
		if (0 <= i%d) && (i%d < (d - k)) {
			U[k][i] = 1
		} else {
			U[k][i] = 0
		}
		if (d-k) <= (i%d) && (i%d) < d {
			U[k-d][i] = 1
		} else {
			U[k-d][i] = 0
		}
	}
	return
}

func Gen_colShift_diagonalVectors_batch(d int, k int, g int) (U map[int][]float64, err error) {
	if k < 0 || k >= d || d <= 0 { //  k < 1 || k >= d || d <= 0
		return nil, errors.New("dimension d <= 0 or k < 1 or k >= d")
	}
	if g <= 0 {
		return nil, errors.New("batchsize g <= 0")
	}
	U = make(map[int][]float64, 2)
	U[k*g] = make([]float64, d*d*g)
	U[(k-d)*g] = make([]float64, d*d*g)
	for i := 0; i < d*d; i++ {
		if (0 <= i%d) && (i%d < (d - k)) {
			for b := 0; b < g; b++ {
				U[k*g][i*g+b] = 1
			}
		} //else {
		if (d-k) <= (i%d) && (i%d) < d {
			for b := 0; b < g; b++ {
				U[(k-d)*g][i*g+b] = 1
			}
		} //else {
	}
	return
}

func Gen_rowShift_diagonalVectors(d int, k int) (U map[int][]float64, err error) {
	if k < 0 || k >= d || d <= 0 { //  k < 1 || k >= d || d <= 0
		return nil, errors.New("dimension d <= 0 or k < 1 or k >= d")
	}
	U = make(map[int][]float64, 1)
	U[k*d] = make([]float64, d*d)
	for i := 0; i < d*d; i++ {
		U[k*d][i] = 1
	}
	return
}

func Gen_transpose_diagonalVectors(d int) (U map[int][]float64, err error) {
	if d <= 0 {
		return nil, errors.New("dimension d <= 0 ")
	}
	U = make(map[int][]float64, 2*d-1)
	/*
		for i := -d + 1; i < d; i++ {
			U[((d-1)*i+d*d)%(d*d)] = make([]float64, d*d)
			for l := 0; l < d*d; l++ {
				k := (l - i) % (d + 1)
				j := (l - i) / (d + 1)
				if i >= 0 {
					if k == 0 && 0 <= j && j < d-i {
						U[((d-1)*i+d*d)%(d*d)][l] = 1
					} else {
						U[((d-1)*i+d*d)%(d*d)][l] = 0
					}
				} else {
					if k == 0 && -i <= j && j < d {
						U[((d-1)*i+d*d)%(d*d)][l] = 1
					} else {
						U[((d-1)*i+d*d)%(d*d)][l] = 0
					}
				}
			}
		}
	*/

	for i := -d + 1; i < d; i++ {
		U[(d-1)*i] = make([]float64, d*d)
		for l := 0; l < d*d; l++ {
			k := (l - i) % (d + 1)
			j := (l - i) / (d + 1)
			if i >= 0 {
				if k == 0 && 0 <= j && j < d-i {
					U[(d-1)*i][l] = 1
				} else {
					U[(d-1)*i][l] = 0
				}
			} else {
				if k == 0 && -i <= j && j < d {
					U[(d-1)*i][l] = 1
				} else {
					U[(d-1)*i][l] = 0
				}
			}
		}
	}

	return
}

func Gen_trans_C_tao_diagonalVectors(d int) (U map[int][]float64, err error) {
	var U_tao map[int][]float64
	var U_trans map[int][]float64
	var sum float64
	U_tao, err = Gen_tao_diagonalVectors(d)
	if err != nil {
		return nil, err
	}
	U_trans, err = Gen_transpose_diagonalVectors(d)
	if err != nil {
		return nil, err
	}
	M_tao := DiagonalVectors2Matrix(U_tao, d*d)
	M_trans := DiagonalVectors2Matrix(U_trans, d*d)
	M := make([][]float64, d*d)
	for i := 0; i < d*d; i++ {
		M[i] = make([]float64, d*d)
	}
	for i := 0; i < d*d; i++ {
		for j := 0; j < d*d; j++ {
			sum = 0
			for k := 0; k < d*d; k++ {
				sum += M_tao[j][k] * M_trans[k][i]
			}
			M[j][i] = sum
		}
	}
	U, err = Matrix2DiagonalVectors(M)
	return
}

func Gen_AlgoA_diagonalVectors(d int, n int) (U map[int][]float64, err error) {
	if d*d*d > n {
		err = errors.New("d^3 > n")
	}
	U_mat := make([][]float64, n)
	for i := range U_mat {
		U_mat[i] = make([]float64, n)
		for j := range U_mat[i] {
			if (i < d*d*d) && (j < d*d) && ((i % d) == 0) && (j == ((i % (d * d)) + int((i / (d * d))))) {
				U_mat[i][j] = 1
			}
		}
	}
	U, err = Matrix2DiagonalVectors(U_mat)
	return
}

func IsInMap(key int, M map[int]interface{}) int {
	_, ok := M[key]
	if ok {
		return 1
	} else {
		return 0
	}
}

func isAllzero(arr []float64) int {
	for i := 0; i < len(arr); i++ {
		if arr[i] != 0 {
			return 0
		}
	}
	return 1
}

func DiagonalVectors2Matrix(U map[int][]float64, n int) (M [][]float64) {
	M = make([][]float64, n)
	for i := 0; i < n; i++ {
		M[i] = make([]float64, n)
	}
	for key, data := range U {
		i := 0
		j := (key + n) % n
		for ; i < n; i++ {
			M[i][j] = data[i]
			j = (j + 1) % n
		}
	}
	return
}

func Matrix2DiagonalVectors(M [][]float64) (U map[int][]float64, err error) {
	if len(M[0]) != len(M) {
		return nil, errors.New("input matrix is not a square one")
	}
	U = make(map[int][]float64, len(M))
	for l := 0; l < len(M); l++ {
		U[l] = make([]float64, len(M))
		j := l
		for i := 0; i < len(M); i++ {
			U[l][i] = M[i][j]
			j = (j + 1) % len(M)
		}
		if isAllzero(U[l]) == 1 {
			delete(U, l)
		}
	}
	return
}

func GenSigmaDiagnalDecomposeMatrices(MatrixDimension int, TargetMaxDiagonalNo int) (Z1s []map[int][]float64, Z2s []map[int][]float64, err error) {
	maxNo := TargetMaxDiagonalNo
	n := MatrixDimension
	n2 := int(math.Ceil(float64(n-1) / 2.0))
	dpNum1 := int(math.Ceil((float64(n-1) / float64(maxNo)))) - 1
	dpNum2 := int(math.Ceil((float64(n2) / float64(maxNo)))) - 1
	Z, err := Gen_sigma_diagonalVecotrs(n)
	if err != nil {
		return nil, nil, err
	}
	Z1 := make(map[int][]float64)
	Z2 := make(map[int][]float64)
	Zl := make(map[int][]float64)
	Zr := make(map[int][]float64)
	Z1s = make([]map[int][]float64, dpNum1+1)
	Z2s = make([]map[int][]float64, dpNum2+1)
	recorder1 := make(map[int]int) // record the key-value: (submatrixNo,nonZeroDiagNo)
	recorder2 := make(map[int]int)

	for key, value := range Z {
		if math.Abs(float64(key)) > math.Ceil(float64(n-1)/2.0) {
			Z1[key] = value
		} else {
			Z2[key] = value
		}
	}

	if (n % 2) == 0 {
		Z1[-int(math.Ceil(float64(n-1)/2.0))] = Z[-int(math.Ceil(float64(n-1)/2.0))]
		delete(Z2, -int(math.Ceil(float64(n-1)/2.0)))
	}

	for i := 0; i < n; i++ {
		if i <= n2 {
			if Z1[i-n] != nil {
				recorder1[i] = i - n
			}
			if Z2[i] != nil {
				recorder2[i] = i
			}

		} else {
			if Z1[i] != nil {
				recorder1[i] = i
			}
			if Z2[i-n] != nil {
				recorder2[i] = i - n
			}
		}
	}

	if dpNum1 == 0 {
		Z1s[0] = DeepIntFloat64MapCopy(Z) // Z
		Z2s = nil
		return
	}
	for i := 0; i < dpNum1; i++ {
		for subMtrxNo, key := range recorder1 { // go through each submatrix.
			value := Z1[key]
			if value == nil {
				continue
			}
			keyAbs := int(math.Abs(float64(key)))
			keySign := 1
			if key != 0 {
				keySign = key / keyAbs
			}
			var jl int
			var jr int
			var start int
			var end int
			var startl int
			var endl int
			var startr int
			var endr int
			if keyAbs >= maxNo {
				jl = keySign * maxNo
				jr = key - keySign*maxNo
			} else {
				jl = 0
				jr = key
			}
			if Zl[jl] == nil {
				Zl[jl] = make([]float64, n*n)
			}
			if Zr[jr] == nil {
				Zr[jr] = make([]float64, n*n)
			}

			if keySign == -1 {
				distance := -key + jr
				start = subMtrxNo*n + keyAbs
				end = (subMtrxNo + 1) * n
				startr = start - distance
				endr = end - distance
				startl = start
				endl = end
			} else {
				distance := key - jr
				start = subMtrxNo * n
				end = start + (n - key)
				startr = start + distance
				endr = end + distance
				startl = start
				endl = end
			}
			copy(Zr[jr][startr:endr], value[start:end])
			copy(Zl[jl][startl:endl], value[start:end])
			recorder1[subMtrxNo] = jr
		}
		Z1s[i] = DeepIntFloat64MapCopy(Zl) //Zl
		if i == dpNum1-1 {
			Z1s[i+1] = DeepIntFloat64MapCopy(Zr) //Zr
		} else {
			Z1 = DeepIntFloat64MapCopy(Zr) //Zr
		}

		for k := range Zl {
			delete(Zl, k)
		}
		for k := range Zr {
			delete(Zr, k)
		}
	}

	if dpNum2 == 0 {
		Z2s[0] = DeepIntFloat64MapCopy(Z2) //Z2
		return
	}
	for i := 0; i < dpNum2; i++ {
		for subMtrxNo, key := range recorder2 {
			value := Z2[key]
			if value == nil {
				continue
			}
			keyAbs := int(math.Abs(float64(key)))
			keySign := 1
			if key != 0 {
				keySign = key / keyAbs
			}
			var jl int
			var jr int
			var start int
			var end int
			var startl int
			var endl int
			var startr int
			var endr int
			if keyAbs >= maxNo {
				jl = keySign * maxNo
				jr = key - keySign*maxNo
			} else {
				jl = 0
				jr = key
			}
			if Zl[jl] == nil {
				Zl[jl] = make([]float64, n*n)
			}
			if Zr[jr] == nil {
				Zr[jr] = make([]float64, n*n)
			}

			if keySign == -1 {
				distance := -key + jr
				start = subMtrxNo*n + keyAbs
				end = (subMtrxNo + 1) * n
				startr = start - distance
				endr = end - distance
				startl = start
				endl = end
			} else {
				distance := key - jr
				start = subMtrxNo * n
				end = start + (n - key)
				startr = start + distance
				endr = end + distance
				startl = start
				endl = end
			}
			copy(Zr[jr][startr:endr], value[start:end])
			copy(Zl[jl][startl:endl], value[start:end])
			recorder2[subMtrxNo] = jr
		}
		Z2s[i] = DeepIntFloat64MapCopy(Zl) //Zl
		if i == dpNum2-1 {
			Z2s[i+1] = DeepIntFloat64MapCopy(Zr) //Zr
		} else {
			Z2 = DeepIntFloat64MapCopy(Zr) //Zr
		}
		for k := range Zl {
			delete(Zl, k)
		}
		for k := range Zr {
			delete(Zr, k)
		}
	}
	return
}

func GenSigmaDiagnalDecomposeMatrices_Ver2(MatrixDimension int, TargetMaxDiagonalNo int) (Z1s []map[int][]float64, Z2s []map[int][]float64, err error) {
	maxNo := TargetMaxDiagonalNo
	n := MatrixDimension
	dpNum := int(math.Ceil((float64(n-1) / float64(maxNo)))) - 1
	Z, err := Gen_sigma_diagonalVecotrs(n)
	if err != nil {
		return nil, nil, err
	}
	Z1 := make(map[int][]float64)
	Z2 := make(map[int][]float64)
	Zl := make(map[int][]float64)
	Zr := make(map[int][]float64)
	Z1s = make([]map[int][]float64, dpNum+1)
	Z2s = make([]map[int][]float64, dpNum+1)
	recorder1 := make(map[int]int) // record the key-value: (submatrixNo,nonZeroDiagNo)
	recorder2 := make(map[int]int)
	for key, value := range Z {
		if key >= 0 {
			Z1[key] = value
		} else {
			Z2[key] = value
		}
	}
	for i := 0; i < n; i++ {
		recorder1[i] = i
		if i != 0 {
			recorder2[i] = i - n
		}
	}
	if dpNum == 0 {
		Z1s[0] = DeepIntFloat64MapCopy(Z) // Z
		Z2s = nil
		return
	}
	for i := 0; i < dpNum; i++ {
		for subMtrxNo, key := range recorder1 {
			value := Z1[key]
			if value == nil {
				continue
			}
			var jl int
			var jr int
			var start int
			var end int
			var startl int
			var endl int
			var startr int
			var endr int
			if key >= maxNo {
				jl = key - maxNo
				jr = maxNo
			} else {
				jl = key
				jr = 0
			}
			if Zl[jl] == nil {
				Zl[jl] = make([]float64, n*n)
			}
			if Zr[jr] == nil {
				Zr[jr] = make([]float64, n*n)
			}
			distance := key - jr
			start = subMtrxNo * n
			end = start + (n - key)
			startr = start + distance
			endr = end + distance
			startl = start
			endl = end
			copy(Zr[jr][startr:endr], value[start:end])
			copy(Zl[jl][startl:endl], value[start:end])
			recorder1[subMtrxNo] = jl
		}
		Z1s[dpNum+1-i-1] = DeepIntFloat64MapCopy(Zr)
		if i == dpNum-1 {
			Z1s[0] = DeepIntFloat64MapCopy(Zl)
		} else {
			Z1 = DeepIntFloat64MapCopy(Zl)
		}
		for k := range Zl {
			delete(Zl, k)
		}
		for k := range Zr {
			delete(Zr, k)
		}
	}
	for i := 0; i < dpNum; i++ {
		for subMtrxNo, key := range recorder2 {
			value := Z2[key]
			if value == nil {
				continue
			}
			keyAbs := int(math.Abs(float64(key)))
			keySign := -1
			if key != 0 {
				keySign = key / keyAbs
			}
			var jl int
			var jr int
			var start int
			var end int
			var startl int
			var endl int
			var startr int
			var endr int
			if keyAbs >= maxNo {
				jl = key - keySign*maxNo
				jr = keySign * maxNo
			} else {
				jl = key
				jr = 0
			}
			if Zl[jl] == nil {
				Zl[jl] = make([]float64, n*n)
			}
			if Zr[jr] == nil {
				Zr[jr] = make([]float64, n*n)
			}
			distance := -key + jr
			start = subMtrxNo*n + keyAbs
			end = (subMtrxNo + 1) * n
			startr = start - distance
			endr = end - distance
			startl = start
			endl = end
			copy(Zr[jr][startr:endr], value[start:end])
			copy(Zl[jl][startl:endl], value[start:end])
			recorder2[subMtrxNo] = jl
		}
		Z2s[dpNum+1-i-1] = DeepIntFloat64MapCopy(Zr)
		if i == dpNum-1 {
			Z2s[0] = DeepIntFloat64MapCopy(Zl)
		} else {
			Z2 = DeepIntFloat64MapCopy(Zl)
		}
		for k := range Zl {
			delete(Zl, k)
		}
		for k := range Zr {
			delete(Zr, k)
		}
	}
	return

}

func GenTauDiagonalDecomposeMatrices(MatrixDimension int, TargetMaxDiagonalNo int) (Ts []map[int][]float64, err error) {
	maxNo := TargetMaxDiagonalNo
	n := MatrixDimension
	if maxNo%n != 0 {
		return nil, errors.New("targetMaxDiagonalNois not divisible by MatrixDimension")
	}
	dpNum1 := int(math.Ceil((float64(n-1) / float64(maxNo/n)))) - 1
	T, err := Gen_tao_diagonalVectors(n)
	if err != nil {
		return nil, err
	}
	Tl := make(map[int][]float64)
	Tr := make(map[int][]float64)
	Ts = make([]map[int][]float64, dpNum1+1)
	if dpNum1 == 0 {
		Ts[0] = DeepIntFloat64MapCopy(T)
		return Ts, err
	}
	for t := 0; t < dpNum1; t++ {
		for i := 0; i <= maxNo/n; i++ {
			key := i * n
			value := T[key]
			if value == nil {
				continue
			}
			if Tl[0] == nil {
				Tl[0] = make([]float64, n*n)
			}
			if Tr[key] == nil {
				Tr[key] = make([]float64, n*n)
			}
			for idx, entry := range value {
				if entry == 0 {
					continue
				}
				Tl[0][idx] = 1
				Tr[key][idx] = 1
			}
		}
		for i := maxNo/n + 1; i < n; i++ {
			key := i * n
			value := T[key]
			fixkey := maxNo
			if value == nil {
				continue
			}
			if Tl[fixkey] == nil {
				Tl[fixkey] = make([]float64, n*n)
			}
			if Tr[key-fixkey] == nil {
				Tr[key-fixkey] = make([]float64, n*n)
			}
			for idx, entry := range value {
				if entry == 0 {
					continue
				}
				Tl[fixkey][idx] = 1
				Tr[key-fixkey][(key+idx-(key-fixkey)+n*n)%(n*n)] = 1
			}
		}
		Ts[t] = DeepIntFloat64MapCopy(Tl)
		if t == dpNum1-1 {
			Ts[t+1] = DeepIntFloat64MapCopy(Tr)
		} else {
			T = DeepIntFloat64MapCopy(Tr)
		}
		for k := range Tl {
			delete(Tl, k)
		}
		for k := range Tr {
			delete(Tr, k)
		}

	}
	return
}

func GenTauDiagonalDecomposeMatrices_Ver2(MatrixDimension int, TargetMaxDiagonalNo int) (Ts []map[int][]float64, err error) {
	maxNo := TargetMaxDiagonalNo
	n := MatrixDimension
	if maxNo%n != 0 {
		return nil, errors.New("targetMaxDiagonalNois not divisible by MatrixDimension")
	}
	dpNum1 := int(math.Ceil((float64(n-1) / float64(maxNo/n)))) - 1
	T, err := Gen_tao_diagonalVectors(n)
	if err != nil {
		return nil, err
	}
	Tl := make(map[int][]float64)
	Tr := make(map[int][]float64)
	Ts = make([]map[int][]float64, dpNum1+1)
	if dpNum1 == 0 {
		Ts[0] = DeepIntFloat64MapCopy(T)
		return Ts, err
	}
	for t := 0; t < dpNum1; t++ {
		for i := 0; i <= maxNo/n; i++ {
			key := i * n
			value := T[key]
			if value == nil {
				continue
			}
			if Tl[key] == nil {
				Tl[key] = make([]float64, n*n)
			}
			if Tr[0] == nil {
				Tr[0] = make([]float64, n*n)
			}
			for idx, entry := range value {
				if entry == 0 {
					continue
				}
				Tl[key][idx] = 1
				Tr[0][(idx+key+n*n)%(n*n)] = 1
			}
		}
		for i := maxNo/n + 1; i < n; i++ {
			key := i * n
			value := T[key]
			fixkey := maxNo
			if value == nil {
				continue
			}
			if Tl[key-fixkey] == nil {
				Tl[key-fixkey] = make([]float64, n*n)
			}
			if Tr[fixkey] == nil {
				Tr[fixkey] = make([]float64, n*n)
			}
			for idx, entry := range value {
				if entry == 0 {
					continue
				}
				Tl[key-fixkey][idx] = 1
				Tr[fixkey][(key+idx-(fixkey)+n*n)%(n*n)] = 1
			}
		}
		Ts[dpNum1+1-t-1] = DeepIntFloat64MapCopy(Tr)
		if t == dpNum1-1 {
			Ts[0] = DeepIntFloat64MapCopy(Tl)
		} else {
			T = DeepIntFloat64MapCopy(Tl)
		}
		for k := range Tl {
			delete(Tl, k)
		}
		for k := range Tr {
			delete(Tr, k)
		}

	}
	return
}

func Converge2DiagonalDecompose_Sigma(A [][]float64) (D []map[int][]float64, err error) {
	if len(A) != len(A[0]) {
		return nil, errors.New("input matrix is not a square one")
	}
	n := int(math.Sqrt(float64(len(A))))
	var U map[int][]float64
	U, err = Matrix2DiagonalVectors(A)
	if err != nil {
		return nil, err
	}

	diagVecs := make([]int, 0)
	diagVecs = append(diagVecs, 0)
	for i := 1; i <= int(math.Ceil(float64(n-1)/2.0)); i++ {
		diagVecs = append(diagVecs, i)
		diagVecs = append(diagVecs, -i+n*n)
	}

	D = make([]map[int][]float64, 2)
	D[0] = make(map[int][]float64)
	D[1] = make(map[int][]float64)
	for _, i := range diagVecs {
		D[0][i] = make([]float64, n*n)
		D[1][i] = make([]float64, n*n)
	}

	for i := -int(math.Ceil(float64(n-1) / 2.0)); i <= int(math.Ceil(float64(n-1)/2.0)); i++ {
		if i == 0 {
			continue
		}
		var j int
		if i > 0 {
			j = (i - n + n*n) % (n * n) // i>0, j<0 i-j = n
		} else {
			j = (i + n + n*n) % (n * n) // i<0, j>0 j-i = n
		}

		j1, j2, ok := findSums(diagVecs, j, n*n)
		if !ok {
			return nil, errors.New("cannot find sums")
		}
		for l, value := range U[j] {
			if value != 1 {
				continue
			}
			D[0][j1][(j+l-j1+n*n)%(n*n)] = 1
			D[1][j2][l] = 1
			z1 := (j + l - j1 + n*n) % (n * n)
			z2 := (j + l - j1 + n*n) % (n * n)
			for z1 != -1 {
				if U[(i+n*n)%(n*n)][z1] == 1 {
					D[0][0][(i+z1+n*n)%(n*n)] = 1
					D[1][(i+n*n)%(n*n)][z1] = 1
					z1 = (i + z1 + n*n) % (n * n)
				} else {
					z1 = -1
				}
			}
			for z2 != -1 {
				if U[(i+n*n)%(n*n)][(z2-i+n*n)%(n*n)] == 1 {
					D[0][(i+n*n)%(n*n)][(z2-i+n*n)%(n*n)] = 1
					D[1][0][(z2-i+n*n)%(n*n)] = 1
					z2 = (z2 - i + n*n) % (n * n)
				} else {
					z2 = -1
				}
			}

		}
		/*
			for l, value := range U[i] {
				if value != 1 {
					continue
				}
				occupied := false
				for _, A1diagVec := range D[0] {
					if A1diagVec[l] == 1 {
						occupied = true
						break
					}
				}
				if occupied {
					D[0][0][(i+l+n*n)%(n*n)] = 1
					D[1][i][l] = 1
				} else {
					D[0][i][l] = 1
					D[1][0][l] = 1
				}
			}
		*/
	}
	for l, value := range U[0] {
		if value != 1 {
			continue
		}
		D[0][0][l] = 1
		D[1][0][l] = 1
	}
	return
}

func Converge2DiagonalDecompose_Tao(T [][]float64) (D []map[int][]float64, err error) {
	if len(T) != len(T[0]) {
		return nil, errors.New("input matrix is not a square one")
	}
	n := int(math.Sqrt(float64(len(T))))
	var U map[int][]float64
	U, err = Matrix2DiagonalVectors(T)
	if err != nil {
		return nil, err
	}

	diagVecs := make([]int, 0)
	diagVecs = append(diagVecs, 0)
	for i := 1; i <= int(math.Ceil(float64(n-1)/2.0)); i++ {
		diagVecs = append(diagVecs, i*n)
	}

	D = make([]map[int][]float64, 2)
	D[0] = make(map[int][]float64)
	D[1] = make(map[int][]float64)
	for _, i := range diagVecs {
		D[0][i] = make([]float64, n*n)
		D[1][i] = make([]float64, n*n)
	}
	for i := 0; i <= int(math.Ceil(float64(n-1)/2.0)); i++ {
		for l, value := range U[i*n] {
			if value != 1 {
				continue
			}
			D[0][0][(i*n+l+n*n)%(n*n)] = 1
			D[1][i*n][l] = 1
		}
	}
	for i := int(math.Ceil(float64(n-1)/2.0)) + 1; i < n; i++ {
		i1n, i2n, ok := findSums(diagVecs, i*n, n*n)
		if !ok {
			return nil, errors.New("cannot find sums")
		}
		for l, value := range U[i*n] {
			if value != 1 {
				continue
			}
			D[0][i1n][(i*n+l-i1n+n*n)%(n*n)] = 1
			D[1][i2n][l] = 1
		}
	}

	return

}

func Converge2DiagonalDecompose(M [][]float64) (D []map[int][]float64, err error) {
	if len(M) != len(M[0]) {
		return nil, errors.New("input matrix is not a square one")
	}
	/*
		if diagonalvecNum >= len(M) {
			return nil, errors.New("not converge can be done since the required Number of diagonalvectors is too big")
		}
		if diagonalvecNum%2 == 0 {
			diagonalvecNum += 1
		}
	*/

	d := len(M)
	var U map[int][]float64
	U, err = Matrix2DiagonalVectors(M)
	if err != nil {
		return nil, err
	}
	keys := make([]int, 0, len(U))
	for k := range U {
		keys = append(keys, k)
	}
	subkeys := FindMinCompositeDiagVecSet(keys, d)

	D = make([]map[int][]float64, 2)
	D[0] = make(map[int][]float64)
	D[1] = make(map[int][]float64)
	for _, i := range subkeys {
		D[0][i] = make([]float64, d)
		D[1][i] = make([]float64, d)
	}

	subkeysMapNoZero := make(map[int]bool)
	for _, sk := range subkeys {
		if sk != 0 {
			subkeysMapNoZero[sk] = true
		}
	}
	subkeysNoZero := getKeys(subkeysMapNoZero)
	Opt1keysMap := make(map[int]bool)
	Opt2keysMap := make(map[int]bool)
	for _, s := range keys {
		s1, s2, ok := findSums(subkeysNoZero, s, d)
		if s1 == s2 && ok {
			Opt1keysMap[s] = true
		} else {
			if s == 0 && !ok {
				Opt1keysMap[s] = true
			} else {
				Opt2keysMap[s] = true
			}
		}
	}
	opt1keys := getKeys(Opt1keysMap)
	opt2keys := getKeys(Opt2keysMap)
	for _, r := range opt1keys {
		s1, s2, ok := findSums(subkeys, r, d)
		if !ok {
			return nil, err // debugtest
		}
		for i := 0; i < d; i++ {
			if U[r][i] == 1 {
				c := (r + i + d) % d
				l1 := (c - s1 + d) % d
				D[0][s1][l1] = 1
				D[1][s2][i] = 1
			}
		}
	}
	for _, r := range opt2keys {
		s1, s2, ok := findSums(subkeys, r, d)
		if !ok {
			return nil, err // debugtest
		}
		for i := 0; i < d; i++ {
			if U[r][i] == 1 {
				occupied := false
				c := (r + i + d) % d
				l1 := (c - s1 + d) % d
				for _, sk := range subkeys {
					if D[0][sk][l1] == 1 {
						occupied = true
						break
					}
				}
				if !occupied {
					D[0][s1][l1] = 1
					D[1][s2][i] = 1
				} else {
					c := (r + i + d) % d
					l2 := (c - s2 + d) % d
					D[0][s2][l2] = 1
					D[1][s1][i] = 1
				}

			}
		}
	}

	/*
		for _, r := range restkeys {
			s1, s2, ok := findSums(subkeys, r, d)
			if !ok {
				return nil, err // debugtest
			}
			for i := 0; i < d; i++ {
				if U[r][i] == 1 {
					c := (r + i + d) % d
					l1 := (c - s1 + d) % d
					D[0][s1][l1] = 1
					D[1][s2][i] = 1
				}
			}
		}
		for _, k := range subkeys {
			for i := 0; i < d; i++ {
				if U[k][i] == 1 {
					occupied := false
					for _, k1 := range subkeys {
						if D[0][k1][i] == 1 {
							occupied = true
							break
						}
					}
					if occupied { // FIXME: Should implement kick out mechanism.
						c := (k + i + d) % d
						D[0][0][c] = 1
						D[1][k][i] = 1
					} else {
						D[0][k][i] = 1
						D[1][0][i] = 1
					}
				}
			}
		}
	*/

	/* Version 1, deprecated.

	keys := make([]int, 0, len(D[0]))
	for k := range D[0] {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	for i := 0; i < d; i++ {
		idx := WhereistheOne(M[i])
		if idx != -1 && idx != i {
			occupied := false
			for k := 0; k < len(keys); k++ {
				keys[k] = (idx-keys[k]+d)%d - i
			}
			sort.Slice(keys, func(i, j int) bool {
				return math.Abs(float64(keys[i])) < math.Abs(float64(keys[j]))
			})

			for k := 0; k < len(keys); k++ {
				keys[k] = (-(keys[k] + i) + idx + d) % d
			}
			for _, k := range keys {
				occupied = false
				mapidx := (idx - k + d) % d // mapidx = (idx-k) mod d, compute the l=mapidx in diagonalCoordinate (k,l) of D[0]
				if D[0][k][mapidx] == 0 {
					for k2, _ := range D[0] {
						if D[0][k2][mapidx] != 0 {
							occupied = true
						}
					}
					if occupied {
						continue
					}
					key := (mapidx - i + d) % d // compute the diagonalCoordinate (k=key,l=no) of D[1]
					_, ok := D[1][key]
					if ok == false {
						occupied = true
						continue
					}
					no := (mapidx - key + d) % d
					for k2, _ := range D[1] {
						if D[1][k2][no] != 0 {
							occupied = true
						}
					}
					if occupied {
						continue
					}
					if D[1][key][no] == 0 {
						D[0][k][mapidx] = 1
						D[1][key][no] = 1
						break
					}
				}
			}
			if occupied {
				return D, errors.New("can not decompose the matrix.")
			}
		}
	}

	*/

	return
}

func WhereistheOne(Vec []float64) (idx int) {
	idx = -1
	for i := 0; i < len(Vec); i++ {
		if Vec[i] == 1 {
			idx = i
			return
		}
	}
	return
}

func FindMinCompositeDiagVecSet(SA []int, n int) []int {

	S := make(map[int]bool)
	S[0] = true
	R := make(map[int]bool)
	for _, a := range SA {
		if a != 0 {
			R[a] = true
		}
	}
	for len(R) > 0 {
		r := getAnyKey(R)
		delete(R, r)

		found := false
		for s1 := range S {
			target := (r - s1 + n) % n
			if S[target] {
				found = true
				break
			}
		}

		if !found {
			S[r] = true
		}
	}
	Rkeys := difference(SA, getKeys(S))
	delete(S, 0)
	Skeys := getKeys(S)
	for _, key := range Skeys {
		_, _, ok := findSums(getKeys(S), key, n)
		if ok {
			delete(S, key)
			for _, rkey := range Rkeys {
				_, _, okk := findSums(getKeys(S), rkey, n)
				if !okk {
					S[key] = true
					break
				}
			}
		}
	}
	S[0] = true

	/* Greedy Version
	S := make(map[int]bool)
	S[0] = true
	R := make(map[int]bool)
	for _, a := range SA {
		if a != 0 {
			R[a] = true
		}
	}
	for len(R) > 0 {
		r := getAnyKey(R)
		delete(R, r)

		found := false
		for s1 := range S {
			target := (r - s1 + n) % n
			if S[target] {
				found = true
				break
			}
		}

		if !found {
			S[r] = true
		}
	}
	*/

	return getKeys(S)
}

func getAnyKey(m map[int]bool) int {
	for k := range m {
		return k
	}
	return -1
}

func getKeys(m map[int]bool) []int {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func difference(a, b []int) []int {
	m := make(map[int]bool)
	for _, v := range b {
		m[v] = true
	}

	var result []int
	for _, v := range a {
		if !m[v] {
			result = append(result, v)
		}
	}

	return result
}

func findSums(nums []int, s int, n int) (int, int, bool) {
	m := make(map[int]bool, len(nums))
	for i := 0; i < len(nums); i++ {
		m[nums[i]] = true
	}
	for _, num := range nums {
		complement := (s - num + n) % n
		if m[complement] {
			return num, complement, true
		}
	}
	return 0, 0, false
}

func SlowPlainMatrixMult(A [][]float64, B [][]float64, d int) (C [][]float64) {
	var sum float64
	C = make([][]float64, d)
	for i := 0; i < d; i++ {
		C[i] = make([]float64, d)
	}
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			sum = 0
			for k := 0; k < d; k++ {
				sum += A[j][k] * B[k][i]
			}
			C[j][i] = sum
		}
	}
	return
}

func SlowPlainMatrixAdd(A [][]float64, B [][]float64, d int) (C [][]float64) {
	C = make([][]float64, d)
	for i := 0; i < d; i++ {
		C[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			C[i][j] = A[i][j] + B[i][j]
		}
	}
	return
}

func MatrixCompare(A [][]float64, B [][]float64) (equal bool) {
	equal = true
	if len(A) != len(B) {
		equal = false
		return
	} else if len(A[0]) != len(B[0]) {
		equal = false
		return
	}
	for i := 0; i < len(A); i++ {
		for j := 0; j < len(A[0]); j++ {
			if A[i][j] != B[i][j] {
				equal = false
				fmt.Printf("Diag: %d, Row: %d", j-i, i)
				return
			}
		}
	}
	return
}

func DeepIntFloat64MapCopy(src map[int][]float64) (dst map[int][]float64) {
	dst = make(map[int][]float64)
	for key, value := range src {
		copiedSlice := make([]float64, len(value))
		copy(copiedSlice, value)
		dst[key] = copiedSlice
	}
	return
}

func RecognizeSeq(vec []float64, d2 int, seq_map map[int]int) (err error) {
	dim := len(vec)
	for i := 0; i < d2; i++ {
		previous := false
		first_pos := 0
		for j := 0; (j < d2) && (i+j*d2 < dim); j++ {
			current_pos := i + j*d2
			if vec[current_pos] != 0 {
				if previous == false {
					_, exist := seq_map[current_pos]
					if exist {
						panic(errors.New("find conflict in the map while adding new item"))
					} else {
						seq_map[current_pos] = 1
						first_pos = current_pos
						previous = true
					}
				} else if previous == true {
					seq_map[first_pos] += 1
				}
			} else {
				previous = false
			}
		}
	}

	/*
		for i := 0; i < len(vec); i++ {
			if vec[i] != 0 {
				count := 1
				for j := i + d2; j < len(vec); j += d2 {
					if vec[j] != 0 {
						count += 1
					} else {
						_, exist := seq_map[i]
						if exist {
							err = errors.New("find conflict in the map while adding new item")
							return
						} else {
							seq_map[i] = count
						}
						break
					}
				}
			}
		}
	*/
	return
}

func ReorganizeSeq(seq_map map[int]int, d2 int) (err error) {
	seq_vec := make([]float64, 0)
	dst_seq_map := make(map[int]int)
	for a, b := range seq_map {
		for i := 0; i < b; i++ {
			seq_vec = append(seq_vec, float64(a+i*d2))
		}
	}
	err = RecognizeSeq(seq_vec, d2, dst_seq_map)
	seq_map = dst_seq_map
	return
}

func MappingVecIdx2DiagIdx() {

}

func MappingDiagIdx2VecIdx(DiagIdx int, d0 int, maxoffset int) (VecIdx int) {
	if DiagIdx%d0 != 0 {
		panic(errors.New("diag index is not devisible by d0."))
	}
	return DiagIdx/d0 + maxoffset
}

func CheckConflictInDecomp(mat map[int][]float64, d0 int, d2 int, dim int, level int) (mat_decomp []map[int][]float64, err error) {

	b := (len(mat) - 1) / 2 // b originally denotes the number of non-zero diags on one side of mat.

	diagL_map := make([]map[int]int, len(mat))

	diagR_map := make([]map[int][]int, 3)
	for i := 0; i < len(diagR_map); i++ {
		diagR_map[i] = make(map[int][]int)
	}
	diagR_map_E := make([]map[int]int, 3)
	for i := 0; i < len(diagR_map); i++ {
		diagR_map_E[i] = make(map[int]int)
	}

	mat_decomp = make([]map[int][]float64, level+1)

	RC := make([]int, 3)
	RC[0] = 0 // That is, RC[1] = rc, RC[2] = -rc, which will be set differently in each iteration.

	for k, diag := range mat {
		if k%d0 != 0 {
			err = errors.New("diagonal index is not divisible by d0")
			panic(err)
		}
		vec_idx := MappingDiagIdx2VecIdx(k, d0, b)
		diagL_map[vec_idx] = make(map[int]int)
		err = RecognizeSeq(diag, d2, diagL_map[vec_idx])
		if err != nil {
			panic(err)
		}
	}

	rc := b * d0
	if (rc/d0)%2 == 1 {
		rc += d0 // fix rc to be divisible by 2, this has no effect on the performance.
	}

	for depth := 1; depth <= level; depth++ {

		if depth > 1 {
			b = rc / d0
		}

		if (rc/d0)%2 == 1 {
			rc += d0
		}
		rc = rc / 2
		RC[1] = rc
		RC[2] = -rc

		for k := -b; k <= b; k++ {
			idx := k + b
			if k == 0 {
				err = ConvertR_diag(k*d0, diagL_map[idx], 0, diagR_map[0])
			} else if k > 0 {
				err = ConvertR_diag(k*d0, diagL_map[idx], rc, diagR_map[1])
			} else {
				err = ConvertR_diag(k*d0, diagL_map[idx], -rc, diagR_map[2])
			}
			if err != nil {
				panic(err)
			}
		}
		for idx := range diagR_map {
			for a, b := range diagR_map[idx] {
				ae := a + (b[0]-1)*d2
				diagR_map_E[idx][ae] = a
			}
		}

		/*
			nonempty := make([]map[int]bool, 3)
			for idx, diag := range diagR_map {
				for a, b := range diag {
					for i := 0; i < b[0]; i++ {
						nonempty[idx][((a+i*d2)%dim+dim)%dim] = true
					}
				}
			}
		*/

		for a1, b1 := range diagR_map[1] {
			for a2, b2 := range diagR_map[2] {
				conflictNum := CheckConflict(a1, b1[0], a2, b2[0], dim, d2)
				if conflictNum > 0 {

					if b1[1] > rc && b2[1] < -rc {
						err = errors.New("conflict happens between diags greater than rc, cannot resolve")
						panic(err)
					}
					if b1[1] < rc {
						var idx int
						var idx0 int
						var a_new1 int
						var a_new2 int
						if a1 > a2 {
							idx = a1
						} else {
							idx = a1 + (b1[0]-conflictNum)*d2
						}
						idx0 = idx + rc

						diagR_map[0][idx0] = make([]int, 2)
						diagR_map[0][idx0][0] = conflictNum
						diagR_map[0][idx0][1] = b1[1]
						if idx == a1 {
							a_new1 = a1 + conflictNum*d2
							diagR_map[1][a_new1] = make([]int, 2)
							diagR_map[1][a_new1][0] = b1[0] - conflictNum
							diagR_map[1][a_new1][1] = b1[1]
							delete(diagR_map[1], a1)
						} else {
							a_new2 = a2 + conflictNum*d2
							diagR_map[1][a1][0] = a2 - a1
							if b1[0] > (a2-a1)+conflictNum {
								diagR_map[1][a_new2] = make([]int, 2)
								diagR_map[1][a_new2][0] = b1[0] - (a2 - a1) - conflictNum
							}
						}
						ConflictZero2RC(diagR_map, idx+rc, diagR_map[0][idx0], dim, d2, rc)
					} else if b2[1] > -rc {
						var idx int
						var idx0 int
						var a_new1 int
						var a_new2 int
						if a2 > a1 {
							idx = a2
						} else {
							idx = a2 + (b2[0]-conflictNum)*d2
						}
						idx0 = idx + rc

						diagR_map[0][idx0] = make([]int, 2)
						diagR_map[0][idx0][0] = conflictNum
						diagR_map[0][idx0][1] = b2[1]
						if idx == a2 {
							a_new1 = a2 + conflictNum*d2
							diagR_map[2][a_new1] = make([]int, 2)
							diagR_map[2][a_new1][0] = b2[0] - conflictNum
							diagR_map[2][a_new1][1] = b2[1]
							delete(diagR_map[2], a2)
						} else {
							a_new2 = a1 + conflictNum*d2
							diagR_map[2][a2][0] = a1 - a2
							if b2[0] > (a1-a2)+conflictNum {
								diagR_map[2][a_new2] = make([]int, 2)
								diagR_map[2][a_new2][0] = b2[0] - (a1 - a2) - conflictNum
							}
						}
						ConflictZero2RC(diagR_map, idx0, diagR_map[0][idx0], dim, d2, -rc)
					} else {
						err = errors.New("cannot resolve...")
						panic(err)
					}

				}
			}

		}
		diagL_map = make([]map[int]int, 2*rc/d0+1) // At this point, we manage to decomposed diags on A_L in range [-rc,rc],
		for idx, diag := range diagR_map {
			for a, b := range diag {
				k := b[1]
				l := a + RC[idx] - k
				new_diag_idx := k - RC[idx] + rc/d0 // recall that the mapping for idx is: idx = k + 2*rc/d0, Here since the scale of A_L
				diagL_map[new_diag_idx][l] = b[0]
			}
		}
		for k, _ := range diagL_map {
			err = ReorganizeSeq(diagL_map[k], d2)
			if err != nil {
				panic(err)
			}
		}
		mat_decomp[level+1-depth] = ExpandSeq2Vec_Right(diagR_map, d2, dim, rc)
		if depth == level {
			mat_decomp[0], err = ExpandSeq2Vec_Left(diagL_map, mat, d2, d0, dim, rc/d0)
		}

	}
	return
}

func ConflictRC2Zero(diagR_map []map[int][]int, a1 int, b1 []int, dim int, d2 int, rc int) (failnum int) {
	return
}

func ConflictZero2RC(diagR_map []map[int][]int, a1 int, b1 []int, dim int, d2 int, rc int) (failnum int) {
	var idx int
	if rc > 0 {
		idx = 1
	} else {
		idx = 2
	}
	for i := 0; i < b1[0]; i++ {
		a2 := a1 + i*d2
		b2, exists := diagR_map[idx][a2]
		if exists {
			cNum := CheckConflict(a1, b1[0], a2, b2[0], dim, d2)
			if cNum > 0 {
				if b2[0] > rc || b2[0] < -rc {
					return cNum
				} else {
					diagR_map[0][a2-rc] = make([]int, 2)
					diagR_map[0][a2-rc][0] = cNum
					diagR_map[0][a2-rc][1] = b2[1]
					if b2[0] > cNum {
						diagR_map[idx][a2+cNum*d2] = make([]int, 2)
						diagR_map[idx][a2+cNum*d2][0] = b2[0] - cNum
						diagR_map[idx][a2+cNum*d2][1] = b2[1]
					}
					delete(diagR_map[idx], a2)
					newcNum := ConflictZero2RC(diagR_map, a2-rc, diagR_map[0][a2-rc], dim, d2, rc)
					if newcNum > 0 {
						if newcNum < cNum {
							diagR_map[0][a2-rc][0] = cNum - newcNum
						} else {
							delete(diagR_map[0], a2-rc)
						}
						diagR_map[idx][a2+(cNum-newcNum)*d2] = make([]int, 2)
						diagR_map[idx][a2+(cNum-newcNum)*d2][0] = diagR_map[idx][a2+cNum*d2][0] + newcNum
						diagR_map[idx][a2+(cNum-newcNum)*d2][1] = diagR_map[idx][a2+cNum*d2][1]
						delete(diagR_map[idx], a2+cNum*d2)
						return newcNum
					}
				}
			}
		}
	}
	/*
		for a2 := a1 - d2; a2 > -rc; a2 -= d2 {
			b2, exists := diagR_map[1][a2]
			if exists {

			}
		}

		for i := 0; i < b[0]; i++ {
			a2 := a1 + i*d2
			b2, exists := diagR_map[2][a2]
			if exists {

			}
		}
		for a2 := a1 - d2; a2 > -rc; a2 -= d2 {
			b2, exists := diagR_map[2][a2]
			if exists {

			}
		}
	*/
	return 0
}

func ConvertR(k int, l int, k1 int) (l1 int) {
	return (k + l - k1)
}

func ConvertR_diag(k int, diag map[int]int, k1 int, dst_diag map[int][]int) (err error) {

	for a, b := range diag {
		l1 := ConvertR(k, a, k1)
		_, exist := dst_diag[l1]
		if exist {
			err = errors.New("find conflict while adding new item to map")
			return
		} else {
			dst_diag[l1] = make([]int, 2)
			dst_diag[l1][0] = b
			dst_diag[l1][1] = k
		}
	}
	return
}

func mergeMaps(map1, map2 map[int]int) (err error) {
	for key, value := range map1 {
		_, exist := map2[key]
		if exist {
			err = errors.New("find conflict while merging maps")
			return
		} else {
			map2[key] = value
		}
	}
	return
}

func CheckConflict(a1 int, b1 int, a2 int, b2 int, dim int, d2 int) (conflictNum int) {
	if a1 > a2 {
		diff := a1 - a2
		var q int
		for q := 0; q < d2; q++ {
			if (diff+q*dim)%d2 == 0 {
				break
			}
		}
		conflictNum = b1 - (diff+q*dim)/d2
		if conflictNum > b2 {
			conflictNum = b2
		}
		if conflictNum < 0 {
			conflictNum = 0
		}
	} else {
		diff := a2 - a1
		var q int
		for q := 0; q < d2; q++ {
			if (diff+q*dim)%d2 == 0 {
				break
			}
		}
		conflictNum = b2 - (diff+q*dim)/d2
		if conflictNum > b1 {
			conflictNum = b1
		}
		if conflictNum < 0 {
			conflictNum = 0
		}
	}
	return
}

func ExtractAndSplit(a int, b int, target int, d2 int) (a1 int, b1 int, a2 int, b2 int, hasfirst bool, hassecond bool) {
	hasfirst = true
	hassecond = true
	a1 = a
	b1 = target - 1 + 1
	if b1 <= 0 {
		hasfirst = false
	}
	a2 = a + (target+1)*d2
	b2 = b - target + 1
	if b2 <= 0 {
		hassecond = false
	}
	return
}

func ExpandSeq2Vec_Left(seq_map []map[int]int, origin_diag_map map[int][]float64, d2 int, d0 int, dim int, offset int) (dst_vec_map map[int][]float64, err error) {
	row_coeff_view := make(map[int]float64)
	dst_vec_map = make(map[int][]float64)

	for _, diag := range origin_diag_map {
		if len(diag) != dim {
			err = errors.New("input origin_diag_map's vector length does not match the matrix dimension")
		}
		for idx, val := range diag {
			row_coeff_view[idx] = val
		}
	}

	for i := range seq_map {
		idx := (i - offset) * d0
		dst_vec_map[idx] = make([]float64, dim)
		for a, b := range seq_map[i] {
			for j := 0; j < b; j++ {
				row_idx := ((a+j*d2)%dim + dim) % dim
				dst_vec_map[idx][row_idx] = row_coeff_view[row_idx]
			}
		}
	}
	return
}

func ExpandSeq2Vec_Right(seq_map []map[int][]int, d2 int, dim int, rc int) (dst_vec_map map[int][]float64) {
	dst_vec_map = make(map[int][]float64)
	RC := make([]int, 3)
	RC[0] = 0
	RC[1] = rc
	RC[2] = -rc
	for i := range seq_map {
		idx := RC[i]
		dst_vec_map[idx] = make([]float64, dim)
		for a, b := range seq_map[i] {
			for j := 0; j < b[0]; j++ {
				row_idx := ((a+j*d2)%dim + dim) % dim
				dst_vec_map[idx][row_idx] = 1
			}
		}
	}
	return
}

func CheckConflictInDecompV2(mat map[int][]float64, d0 int, dim int, level int) (mat_decomp []map[int][]float64, err error) {
	b := (len(mat) - 1) / 2 // b originally denotes the number of non-zero diags on one side of mat.

	diagL_map := make([]map[int]bool, len(mat))

	diagR_map := make([]map[int]int, 3)

	mat_decomp = make([]map[int][]float64, level+1)

	RC := make([]int, 3)
	RC[0] = 0 // That is, RC[1] = rc, RC[2] = -rc, which will be set differently in each iteration.

	for k, diag := range mat {
		if k%d0 != 0 {
			err = errors.New("diagonal index is not divisible by d0")
			panic(err)
		}
		vec_idx := MappingDiagIdx2VecIdx(k, d0, b)
		diagL_map[vec_idx] = make(map[int]bool)
		for idx, val := range diag {
			if val != 0 {
				diagL_map[vec_idx][idx] = true
			}
		}
	}
	rc := b * d0
	if (rc/d0)%2 == 1 {
		rc += d0 // fix rc to be divisible by 2, this has no effect on the performance.
	}

	for depth := 1; depth <= level; depth++ {
		if depth > 1 {
			b = rc / d0
		}

		if (rc/d0)%2 == 1 {
			rc += d0
		}
		rc = rc / 2
		RC[1] = rc
		RC[2] = -rc

		for i := 0; i < len(diagR_map); i++ {
			diagR_map[i] = make(map[int]int, 0)
		}

		for k := -b; k <= b; k++ {
			idx := k + b
			if k == 0 {
				for pos := range diagL_map[idx] {
					diagR_map[0][((ConvertR(k*d0, pos, 0)%dim)+dim)%dim] = k * d0
				}
			} else if k > 0 {
				for pos := range diagL_map[idx] {
					diagR_map[1][((ConvertR(k*d0, pos, rc)%dim)+dim)%dim] = k * d0
				}
			} else {
				for pos := range diagL_map[idx] {
					diagR_map[2][((ConvertR(k*d0, pos, -rc)%dim)+dim)%dim] = k * d0
				}
			}
		}
		for a1, b1 := range diagR_map[1] {
			b2, exists := diagR_map[2][a1]
			if exists {
				solved := false
				if b1 > rc && b2 < -rc {
					err = errors.New("conflict happens between diags greater than rc, cannot resolve")
					panic(err)
				}
				if b1 < rc {
					a0 := ((a1+rc)%dim + dim) % dim
					diagR_map[0][a0] = b1
					delete(diagR_map[1], a1)
					solved = CheckZero2RC(diagR_map, a0, diagR_map[0][a0], dim, rc)
					if !solved {
						diagR_map[1][a1] = diagR_map[0][a0]
						delete(diagR_map[0], a0)
					}
				} else if (b2 > -rc) && (!solved) {
					a0 := ((a1-rc)%dim + dim) % dim
					diagR_map[0][a0] = b2
					delete(diagR_map[2], a1)
					solved = CheckZero2RC(diagR_map, a0, diagR_map[0][a0], dim, rc)
				}
				if !solved {
					panic(errors.New("cannot resolved"))
				}
			}
		}
		diagL_map = make([]map[int]bool, 2*rc/d0+1)
		for idx, diag := range diagR_map {
			for a, b := range diag {
				k := b
				l := ((a+RC[idx]-k)%dim + dim) % dim
				new_diag_idx := (k-RC[idx])/d0 + rc/d0
				if diagL_map[new_diag_idx] == nil {
					diagL_map[new_diag_idx] = make(map[int]bool)
				}
				diagL_map[new_diag_idx][l] = true
			}
		}
		mat_decomp[level+1-depth] = make(map[int][]float64)
		for idx, diag := range diagR_map {
			mat_decomp[level+1-depth][RC[idx]] = make([]float64, dim)
			for a := range diag {
				mat_decomp[level+1-depth][RC[idx]][a] = 1
			}
		}
		if depth == level {
			mat_decomp[0] = make(map[int][]float64)
			for idx, diag := range diagL_map {
				mat_decomp[0][(idx-(rc/d0))*d0] = make([]float64, dim)
				for a := range diag {
					mat_decomp[0][(idx-(rc/d0))*d0][a] = 1
				}
			}
		}
	}

	return
}

func CheckZero2RC(diagR_map []map[int]int, a0 int, b0 int, dim int, rc int) (solved bool) {

	b1, exists1 := diagR_map[1][a0]
	b2, exists2 := diagR_map[2][a0]
	var a0_new int
	solved = true

	if exists1 {
		if b1 > rc {
			return false
		}
		a0_new = ((a0+rc)%dim + dim) % dim
		diagR_map[0][a0_new] = b1
		delete(diagR_map[1], a0)
		solved = CheckZero2RC(diagR_map, a0_new, b1, dim, rc)
		if !solved {
			delete(diagR_map[0], a0_new)
			diagR_map[1][a0] = b1
			solved = false
		}
	} else if exists2 {
		if b2 < -rc {
			return false
		}
		a0_new = ((a0-rc)%dim + dim) % dim
		diagR_map[0][a0_new] = b2
		delete(diagR_map[2], a0)
		solved = CheckZero2RC(diagR_map, a0_new, b2, dim, rc)
		if !solved {
			delete(diagR_map[0], a0_new)
			diagR_map[2][a0] = b2
			solved = false
		}
	}
	return solved
}

func CheckConflictInDecompV3(mat map[int][]float64, d0 int, dim int, level int) (mat_decomp []map[int][]float64, err error) {
	b := (len(mat) - 1) / 2 // b originally denotes the number of non-zero diags on one side of mat.

	diagL_map := make([]map[int]bool, len(mat))

	diagR_map := make([]map[int]int, 3)

	mat_decomp = make([]map[int][]float64, level+1)

	RC := make([]int, 3)
	RC[0] = 0 // That is, RC[1] = rc, RC[2] = -rc, which will be set differently in each iteration.

	for k, diag := range mat {
		if k%d0 != 0 {
			err = errors.New("diagonal index is not divisible by d0")
			panic(err)
		}
		vec_idx := MappingDiagIdx2VecIdx(k, d0, b)
		diagL_map[vec_idx] = make(map[int]bool)
		for idx, val := range diag {
			if val != 0 {
				diagL_map[vec_idx][idx] = true
			}
		}
	}
	rc := b * d0
	if (rc/d0)%2 == 1 {
		rc += d0 // fix rc to be divisible by 2, this has no effect on the performance.
	}

	for depth := 1; depth <= level; depth++ {
		if depth > 1 {
			b = rc / d0
		}

		if (rc/d0)%2 == 1 {
			rc += d0
		}
		rc = rc / 2
		RC[1] = rc
		RC[2] = -rc

		for i := 0; i < len(diagR_map); i++ {
			diagR_map[i] = make(map[int]int, 0)
		}

		for k := -b; k <= b; k++ {
			idx := k + b
			if k == 0 {
				for pos := range diagL_map[idx] {
					diagR_map[0][((ConvertR(k*d0, pos, 0)%dim)+dim)%dim] = k * d0
				}
			} else if k > 0 {
				for pos := range diagL_map[idx] {
					diagR_map[1][((ConvertR(k*d0, pos, rc)%dim)+dim)%dim] = k * d0
				}
			} else {
				for pos := range diagL_map[idx] {
					diagR_map[2][((ConvertR(k*d0, pos, -rc)%dim)+dim)%dim] = k * d0
				}
			}
		}
		for a1, b1 := range diagR_map[1] {
			b2, exists := diagR_map[2][a1]
			if exists {
				solved := false
				if b1 > rc && b2 < -rc {
					err = errors.New("conflict happens between diags greater than rc, cannot resolve")
					panic(err)
				}
				if b1 < rc {
					a0 := ((a1+rc)%dim + dim) % dim
					diagR_map[0][a0] = b1
					delete(diagR_map[1], a1)

					h_onetwo := make([]int, 0)
					h_a0 := make([]int, 0)
					h_a0_new := make([]int, 0)
					h_b := make([]int, 0)

					h_onetwo = append(h_onetwo, 1)
					h_a0 = append(h_a0, a1)
					h_a0_new = append(h_a0_new, a0)
					h_b = append(h_b, b1)

					solved, _ = CheckZero2RCV2(diagR_map, a0, diagR_map[0][a0], dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
					if !solved {
						diagR_map[1][a1] = diagR_map[0][a0]
						delete(diagR_map[0], a0)
					}
				} else if (b2 > -rc) && (!solved) {
					a0 := ((a1-rc)%dim + dim) % dim
					diagR_map[0][a0] = b2
					delete(diagR_map[2], a1)

					h_onetwo := make([]int, 0)
					h_a0 := make([]int, 0)
					h_a0_new := make([]int, 0)
					h_b := make([]int, 0)

					h_onetwo = append(h_onetwo, 2)
					h_a0 = append(h_a0, a1)
					h_a0_new = append(h_a0_new, a0)
					h_b = append(h_b, b2)

					solved, _ = CheckZero2RCV2(diagR_map, a0, diagR_map[0][a0], dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
				}
				if !solved {
					panic(errors.New("cannot resolved"))
				}
			}
		}
		diagL_map = make([]map[int]bool, 2*rc/d0+1)
		for idx, diag := range diagR_map {
			for a, b := range diag {
				k := b
				l := ((a+RC[idx]-k)%dim + dim) % dim
				new_diag_idx := (k-RC[idx])/d0 + rc/d0
				if diagL_map[new_diag_idx] == nil {
					diagL_map[new_diag_idx] = make(map[int]bool)
				}
				diagL_map[new_diag_idx][l] = true
			}
		}
		mat_decomp[level+1-depth] = make(map[int][]float64)
		for idx, diag := range diagR_map {
			mat_decomp[level+1-depth][RC[idx]] = make([]float64, dim)
			for a := range diag {
				mat_decomp[level+1-depth][RC[idx]][a] = 1
			}
		}
		if depth == level {
			mat_decomp[0] = make(map[int][]float64)
			for idx, diag := range diagL_map {
				mat_decomp[0][(idx-(rc/d0))*d0] = make([]float64, dim)
				for a := range diag {
					mat_decomp[0][(idx-(rc/d0))*d0][a] = 1
				}
			}
		}
	}

	return
}

func CheckConflictInDecompV4(mat map[int][]float64, d0 int, dim int, level int) (mat_decomp []map[int][]float64, zero_diag []float64, err error) {
	b := (len(mat) - 1) / 2 // b originally denotes the number of non-zero diags on one side of mat.

	diagL_map := make([]map[int]bool, len(mat))

	diagR_map := make([]map[int]int, 3)

	mat_decomp = make([]map[int][]float64, level+1)

	zero_diag = make([]float64, dim)

	copy(mat[0], zero_diag)

	RC := make([]int, 3)
	RC[0] = 0 // That is, RC[1] = rc, RC[2] = -rc, which will be set differently in each iteration.

	for k, diag := range mat {
		if k%d0 != 0 {
			err = errors.New("diagonal index is not divisible by d0")
			panic(err)
		}
		vec_idx := MappingDiagIdx2VecIdx(k, d0, b)
		diagL_map[vec_idx] = make(map[int]bool)
		for idx, val := range diag {
			if k == 0 {
				val = 0
			}
			if val != 0 {
				diagL_map[vec_idx][idx] = true
			}
		}
	}
	rc := b * d0
	if (rc/d0)%2 == 1 {
		rc += d0 // fix rc to be divisible by 2, this has no effect on the performance.
	}

	for depth := 1; depth <= level; depth++ {
		if depth > 1 {
			b = rc / d0
		}

		if (rc/d0)%2 == 1 {
			rc += d0
		}
		rc = rc / 2
		RC[1] = rc
		RC[2] = -rc

		for i := 0; i < len(diagR_map); i++ {
			diagR_map[i] = make(map[int]int, 0)
		}

		for k := -b; k <= b; k++ {
			idx := k + b
			if k == 0 {
				for pos := range diagL_map[idx] {
					diagR_map[0][((ConvertR(k*d0, pos, 0)%dim)+dim)%dim] = k * d0
				}
			} else if k > 0 {
				for pos := range diagL_map[idx] {
					diagR_map[1][((ConvertR(k*d0, pos, rc)%dim)+dim)%dim] = k * d0
				}
			} else {
				for pos := range diagL_map[idx] {
					diagR_map[2][((ConvertR(k*d0, pos, -rc)%dim)+dim)%dim] = k * d0
				}
			}
		}
		for a1, b1 := range diagR_map[1] {
			b2, exists := diagR_map[2][a1]
			if exists {
				solved := false
				if b1 > rc && b2 < -rc {
					err = errors.New("conflict happens between diags greater than rc, cannot resolve")
					panic(err)
				}
				if b1 < rc {
					a0 := ((a1+rc)%dim + dim) % dim
					diagR_map[0][a0] = b1
					delete(diagR_map[1], a1)

					h_onetwo := make([]int, 0)
					h_a0 := make([]int, 0)
					h_a0_new := make([]int, 0)
					h_b := make([]int, 0)

					h_onetwo = append(h_onetwo, 1)
					h_a0 = append(h_a0, a1)
					h_a0_new = append(h_a0_new, a0)
					h_b = append(h_b, b1)

					solved, _ = CheckZero2RCV2(diagR_map, a0, diagR_map[0][a0], dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
					if !solved {
						diagR_map[1][a1] = diagR_map[0][a0]
						delete(diagR_map[0], a0)
					}
				} else if (b2 > -rc) && (!solved) {
					a0 := ((a1-rc)%dim + dim) % dim
					diagR_map[0][a0] = b2
					delete(diagR_map[2], a1)

					h_onetwo := make([]int, 0)
					h_a0 := make([]int, 0)
					h_a0_new := make([]int, 0)
					h_b := make([]int, 0)

					h_onetwo = append(h_onetwo, 2)
					h_a0 = append(h_a0, a1)
					h_a0_new = append(h_a0_new, a0)
					h_b = append(h_b, b2)

					solved, _ = CheckZero2RCV2(diagR_map, a0, diagR_map[0][a0], dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
				}
				if !solved {
					panic(errors.New("cannot resolved"))
				}
			}
		}
		diagL_map = make([]map[int]bool, 2*rc/d0+1)
		for idx, diag := range diagR_map {
			for a, b := range diag {
				k := b
				l := ((a+RC[idx]-k)%dim + dim) % dim
				new_diag_idx := (k-RC[idx])/d0 + rc/d0
				if diagL_map[new_diag_idx] == nil {
					diagL_map[new_diag_idx] = make(map[int]bool)
				}
				diagL_map[new_diag_idx][l] = true
			}
		}
		mat_decomp[level+1-depth] = make(map[int][]float64)
		for idx, diag := range diagR_map {
			mat_decomp[level+1-depth][RC[idx]] = make([]float64, dim)
			for a := range diag {
				mat_decomp[level+1-depth][RC[idx]][a] = 1
			}
		}
		if depth == level {
			mat_decomp[0] = make(map[int][]float64)
			for idx, diag := range diagL_map {
				mat_decomp[0][(idx-(rc/d0))*d0] = make([]float64, dim)
				for a := range diag {
					mat_decomp[0][(idx-(rc/d0))*d0][a] = 1
				}
			}
		}
	}

	return
}

func CheckConflictInDecomp_SingleLayer(mat map[int][]float64, d0 int, dim int) (mat_decomp []map[int][]float64, mat_decomp1 []map[int][]float64, mat_decomp2 []map[int][]float64, err error) {
	pos_count := 0.0
	neg_count := 0.0
	for key := range mat {
		if key > 0 {
			pos_count++
		} else if key < 0 {
			neg_count++
		}
	}
	b := int(math.Max(pos_count, neg_count))

	diagL_map := make([]map[int]bool, 2*b+1)

	diagR_map := make([]map[int]int, 3)

	diagL_map1 := make([]map[int]bool, len(mat))

	diagR_map1 := make([]map[int]int, 3)

	diagL_map2 := make([]map[int]bool, len(mat))

	diagR_map2 := make([]map[int]int, 3)

	mat_decomp = make([]map[int][]float64, 2)

	mat_decomp1 = make([]map[int][]float64, 2)

	mat_decomp2 = make([]map[int][]float64, 2)

	RC := make([]int, 3)
	RC[0] = 0 // That is, RC[1] = rc, RC[2] = -rc, which will be set differently in each iteration.

	for k, diag := range mat {
		if k%d0 != 0 {
			err = errors.New("diagonal index is not divisible by d0")
			return
		}
		vec_idx := MappingDiagIdx2VecIdx(k, d0, b)
		diagL_map[vec_idx] = make(map[int]bool)
		for idx, val := range diag {
			if k == 0 {
				val = 0
			}
			if val != 0 {
				diagL_map[vec_idx][idx] = true
			}
		}
	}

	rc := b * d0
	if (rc/d0)%2 == 1 {
		rc += d0 // fix rc to be divisible by 2, this has no effect on the performance.
	}
	rc = rc / 2
	RC[1] = rc
	RC[2] = -rc

	for i := 0; i < len(diagR_map); i++ {
		diagR_map[i] = make(map[int]int, 0)
	}
	for i := 0; i < len(diagR_map); i++ {
		diagR_map1[i] = make(map[int]int, 0)
	}
	for i := 0; i < len(diagR_map); i++ {
		diagR_map2[i] = make(map[int]int, 0)
	}

	for k := -b; k <= b; k++ {
		idx := k + b
		if k == 0 {
			for pos := range diagL_map[idx] {
				diagR_map[0][((ConvertR(k*d0, pos, 0)%dim)+dim)%dim] = k * d0
			}
		} else if k > 0 {
			for pos := range diagL_map[idx] {
				row := ((ConvertR(k*d0, pos, rc) % dim) + dim) % dim
				diagR_map[1][row] = k * d0
			}
		} else {
			for pos := range diagL_map[idx] {
				row := ((ConvertR(k*d0, pos, -rc) % dim) + dim) % dim
				diagR_map[2][row] = k * d0
			}
		}
	}
	for a1, b1 := range diagR_map[1] {
		b2, exists := diagR_map[2][a1]
		if exists {
			solved := false
			conflictNum1 := 0
			conflictNum2 := 0
			if b1 > rc && b2 < -rc {
				/*
					_, exists_tmp := diagR_map1[1][a1]
					if !exists_tmp {
						diagR_map1[1][a1] = b1
						delete(diagR_map[1], a1)
					} else {
						diagR_map2[2][a1] = b2
						delete(diagR_map[2], a1)
					}
				*/
				if _, exists_tmp := diagR_map1[1][a1]; !exists_tmp {
					diagR_map1[1][a1] = b1
					delete(diagR_map[1], a1)
				} else if _, exists_tmp := diagR_map1[2][a1]; !exists_tmp {
					diagR_map1[2][a1] = b2
					delete(diagR_map[2], a1)
				} else if _, exists_tmp := diagR_map2[1][a1]; !exists_tmp {
					diagR_map2[1][a1] = b1
					delete(diagR_map[1], a1)
				} else {
					diagR_map2[2][a1] = b2
					delete(diagR_map[2], a1)
				}

				solved = true
			}
			if b1 <= rc { // stricter condition: b1 < rc , loose condition: b1 <= rc
				a0 := ((a1+rc)%dim + dim) % dim
				diagR_map[0][a0] = b1
				delete(diagR_map[1], a1)

				h_onetwo := make([]int, 0)
				h_a0 := make([]int, 0)
				h_a0_new := make([]int, 0)
				h_b := make([]int, 0)

				h_onetwo = append(h_onetwo, 1)
				h_a0 = append(h_a0, a1)
				h_a0_new = append(h_a0_new, a0)
				h_b = append(h_b, b1)

				conflictNum1 = 1

				var conflictNum1_tmp = 0
				solved, conflictNum1_tmp = CheckZero2RCV2(diagR_map, a0, diagR_map[0][a0], dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
				conflictNum1 = conflictNum1 + conflictNum1_tmp

			} else if (b2 >= -rc) && (!solved) { // stricter condition: b2 > rc , loose condition: b2 >= rc
				a0 := ((a1-rc)%dim + dim) % dim
				diagR_map[0][a0] = b2
				delete(diagR_map[2], a1)

				h_onetwo := make([]int, 0)
				h_a0 := make([]int, 0)
				h_a0_new := make([]int, 0)
				h_b := make([]int, 0)

				h_onetwo = append(h_onetwo, 2)
				h_a0 = append(h_a0, a1)
				h_a0_new = append(h_a0_new, a0)
				h_b = append(h_b, b2)

				conflictNum2 = 1
				var conflictNum2_tmp = 0
				solved, conflictNum2_tmp = CheckZero2RCV2(diagR_map, a0, diagR_map[0][a0], dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
				conflictNum2 = conflictNum2 + conflictNum2_tmp
			}
			if !solved {
				a0 := 0

				if (b1 <= rc) && (b2 < -rc) {
					a0 = ((a1+rc)%dim + dim) % dim
					diagR_map[0][a0] = b1
					delete(diagR_map[1], a1)
				} else if (b1 > rc) && (b2 >= -rc) {
					a0 = ((a1-rc)%dim + dim) % dim
					diagR_map[0][a0] = b2
					delete(diagR_map[2], a1)
				} else {
					if conflictNum1 <= conflictNum2 {
						a0 = ((a1+rc)%dim + dim) % dim
						diagR_map[0][a0] = b1
						delete(diagR_map[1], a1)
					} else {
						a0 = ((a1-rc)%dim + dim) % dim
						diagR_map[0][a0] = b2
						delete(diagR_map[2], a1)
					}
				}

				CheckZero2RCV3(diagR_map, diagR_map1, diagR_map2, a0, diagR_map[0][a0], dim, rc)
			}
		}
	}
	diagL_map = make([]map[int]bool, 2*rc/d0+1)
	for idx, diag := range diagR_map {
		for a, b := range diag {
			k := b
			l := ((a+RC[idx]-k)%dim + dim) % dim
			new_diag_idx := (k-RC[idx])/d0 + rc/d0
			if diagL_map[new_diag_idx] == nil {
				diagL_map[new_diag_idx] = make(map[int]bool)
			}
			diagL_map[new_diag_idx][l] = true
		}
	}

	diagL_map1 = make([]map[int]bool, 2*rc/d0+1)
	for idx, diag := range diagR_map1 {
		for a, b := range diag {
			k := b
			l := ((a+RC[idx]-k)%dim + dim) % dim
			new_diag_idx := (k-RC[idx])/d0 + rc/d0
			if diagL_map1[new_diag_idx] == nil {
				diagL_map1[new_diag_idx] = make(map[int]bool)
			}
			diagL_map1[new_diag_idx][l] = true
		}
	}

	diagL_map2 = make([]map[int]bool, 2*rc/d0+1)
	for idx, diag := range diagR_map2 {
		for a, b := range diag {
			k := b
			l := ((a+RC[idx]-k)%dim + dim) % dim
			new_diag_idx := (k-RC[idx])/d0 + rc/d0
			if diagL_map2[new_diag_idx] == nil {
				diagL_map2[new_diag_idx] = make(map[int]bool)
			}
			diagL_map2[new_diag_idx][l] = true
		}
	}

	level := 1
	depth := 1
	mat_decomp[level+1-depth] = make(map[int][]float64)
	for idx, diag := range diagR_map {
		mat_decomp[level+1-depth][RC[idx]] = make([]float64, dim)
		for a := range diag {
			mat_decomp[level+1-depth][RC[idx]][a] = 1
		}
	}
	if depth == level {
		mat_decomp[0] = make(map[int][]float64)
		for idx, diag := range diagL_map {
			mat_decomp[0][(idx-(rc/d0))*d0] = make([]float64, dim)
			for a := range diag {
				mat_decomp[0][(idx-(rc/d0))*d0][a] = 1
			}
		}
	}

	mat_decomp1[level+1-depth] = make(map[int][]float64)
	for idx, diag := range diagR_map1 {
		mat_decomp1[level+1-depth][RC[idx]] = make([]float64, dim)
		for a := range diag {
			mat_decomp1[level+1-depth][RC[idx]][a] = 1
		}
	}
	if depth == level {
		mat_decomp1[0] = make(map[int][]float64)
		for idx, diag := range diagL_map1 {
			mat_decomp1[0][(idx-(rc/d0))*d0] = make([]float64, dim)
			for a := range diag {
				mat_decomp1[0][(idx-(rc/d0))*d0][a] = 1
			}
		}
	}

	mat_decomp2[level+1-depth] = make(map[int][]float64)
	for idx, diag := range diagR_map2 {
		mat_decomp2[level+1-depth][RC[idx]] = make([]float64, dim)
		for a := range diag {
			mat_decomp2[level+1-depth][RC[idx]][a] = 1
		}
	}
	if depth == level {
		mat_decomp2[0] = make(map[int][]float64)
		for idx, diag := range diagL_map2 {
			mat_decomp2[0][(idx-(rc/d0))*d0] = make([]float64, dim)
			for a := range diag {
				mat_decomp2[0][(idx-(rc/d0))*d0][a] = 1
			}
		}
	}

	return
}

func CheckConflictInDecompV5(mat map[int][]float64, d0 int, dim int, level int) (mat_decomp []map[int][]float64, zero_diags [][]float64, err error) {

	pos_count := 0.0
	neg_count := 0.0
	for key := range mat {
		if key > 0 {
			pos_count++
		} else if key < 0 {
			neg_count++
		}
	}
	b := int(math.Max(pos_count, neg_count)) // if both sides have equal number of diags, then b = (len(mat) - 1) / 2

	diagL_map := make([]map[int]bool, 2*b+1) // make([]map[int]bool, len(mat))

	diagR_map := make([]map[int]int, 3)

	mat_decomp = make([]map[int][]float64, level+1)

	zero_diags = make([][]float64, level)
	zero_diags[len(zero_diags)-1] = make([]float64, dim)
	copy(zero_diags[len(zero_diags)-1], mat[0])

	RC := make([]int, 3)
	RC[0] = 0 // That is, RC[1] = rc, RC[2] = -rc, which will be set differently in each iteration.

	for k, diag := range mat {
		if k%d0 != 0 {
			err = errors.New("diagonal index is not divisible by d0")
			panic(err)
		}
		vec_idx := MappingDiagIdx2VecIdx(k, d0, b)
		diagL_map[vec_idx] = make(map[int]bool)
		for idx, val := range diag {
			if k == 0 {
				val = 0
			}
			if val != 0 {
				diagL_map[vec_idx][idx] = true
			}
		}
	}
	rc := b * d0
	if (rc/d0)%2 == 1 {
		rc += d0 // fix rc to be divisible by 2, this has no effect on the performance.
	}

	for depth := 1; depth <= level; depth++ {
		if depth > 1 {
			b = rc / d0
		}

		if (rc/d0)%2 == 1 {
			rc += d0
		}
		rc = rc / 2
		RC[1] = rc
		RC[2] = -rc

		for i := 0; i < len(diagR_map); i++ {
			diagR_map[i] = make(map[int]int, 0)
		}

		for k := -b; k <= b; k++ {
			idx := k + b
			if k == 0 {
				for pos := range diagL_map[idx] {
					diagR_map[0][((ConvertR(k*d0, pos, 0)%dim)+dim)%dim] = k * d0
				}
			} else if k > 0 {
				for pos := range diagL_map[idx] {
					diagR_map[1][((ConvertR(k*d0, pos, rc)%dim)+dim)%dim] = k * d0
				}
			} else {
				for pos := range diagL_map[idx] {
					diagR_map[2][((ConvertR(k*d0, pos, -rc)%dim)+dim)%dim] = k * d0
				}
			}
		}
		for a1, b1 := range diagR_map[1] {
			b2, exists := diagR_map[2][a1]
			if exists {
				solved := false
				if b1 > rc && b2 < -rc {
					err = errors.New("conflict happens between diags greater than rc, cannot resolve")
					panic(err)
				}
				if b1 < rc {
					a0 := ((a1+rc)%dim + dim) % dim
					diagR_map[0][a0] = b1
					delete(diagR_map[1], a1)

					h_onetwo := make([]int, 0)
					h_a0 := make([]int, 0)
					h_a0_new := make([]int, 0)
					h_b := make([]int, 0)

					h_onetwo = append(h_onetwo, 1)
					h_a0 = append(h_a0, a1)
					h_a0_new = append(h_a0_new, a0)
					h_b = append(h_b, b1)

					solved, _ = CheckZero2RCV2(diagR_map, a0, diagR_map[0][a0], dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
					if !solved {
						diagR_map[1][a1] = diagR_map[0][a0]
						delete(diagR_map[0], a0)
					}
				} else if (b2 > -rc) && (!solved) {
					a0 := ((a1-rc)%dim + dim) % dim
					diagR_map[0][a0] = b2
					delete(diagR_map[2], a1)

					h_onetwo := make([]int, 0)
					h_a0 := make([]int, 0)
					h_a0_new := make([]int, 0)
					h_b := make([]int, 0)

					h_onetwo = append(h_onetwo, 2)
					h_a0 = append(h_a0, a1)
					h_a0_new = append(h_a0_new, a0)
					h_b = append(h_b, b2)

					solved, _ = CheckZero2RCV2(diagR_map, a0, diagR_map[0][a0], dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
				}
				if !solved {
					panic(errors.New("cannot resolved"))
				}
			}
		}
		diagL_map = make([]map[int]bool, 2*rc/d0+1)
		for idx, diag := range diagR_map {
			for a, b := range diag {
				k := b
				l := ((a+RC[idx]-k)%dim + dim) % dim
				new_diag_idx := (k-RC[idx])/d0 + rc/d0
				if diagL_map[new_diag_idx] == nil {
					diagL_map[new_diag_idx] = make(map[int]bool)
				}
				diagL_map[new_diag_idx][l] = true
			}
		}

		if depth < level {
			zero_diags[level-1-depth] = make([]float64, dim) // preserve zero axis
			for idx, val := range diagL_map[rc/d0] {
				if val {
					zero_diags[level-1-depth][idx] = 1
				}
			}
			diagL_map[rc/d0] = make(map[int]bool) // implicitly remove zero axis.
		}

		mat_decomp[level+1-depth] = make(map[int][]float64)
		for idx, diag := range diagR_map {
			mat_decomp[level+1-depth][RC[idx]] = make([]float64, dim)
			for a := range diag {
				mat_decomp[level+1-depth][RC[idx]][a] = 1
			}
		}
		if depth == level {
			mat_decomp[0] = make(map[int][]float64)
			for idx, diag := range diagL_map {
				mat_decomp[0][(idx-(rc/d0))*d0] = make([]float64, dim)
				for a := range diag {
					mat_decomp[0][(idx-(rc/d0))*d0][a] = 1
				}
			}
		}
	}

	return
}

func CheckZero2RCV2(diagR_map []map[int]int, a0 int, b0 int, dim int, rc int, h_onetwo []int, h_a0 []int, h_a0_new []int, h_b []int) (solved bool, conflictNum int) {
	b1, exists1 := diagR_map[1][a0]
	b2, exists2 := diagR_map[2][a0]
	var a0_new1 int
	var a0_new2 int
	solved = true
	var solved1 = true
	var solved2 = true
	conflictNum = 0

	if exists1 {
		if b1 > rc {
			solved1 = false
		} else {
			a0_new1 = ((a0+rc)%dim + dim) % dim
			diagR_map[0][a0_new1] = b1
			delete(diagR_map[1], a0)
			h_onetwo = append(h_onetwo, 1)
			h_a0 = append(h_a0, a0)
			h_a0_new = append(h_a0_new, a0_new1)
			h_b = append(h_b, b1)

			conflictNum_tmp := 0
			solved1, conflictNum_tmp = CheckZero2RCV2(diagR_map, a0_new1, b1, dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
			if solved1 {
				conflictNum = conflictNum + conflictNum_tmp + 1
			}
		}
		if !solved1 {
			solved = false

		}
	}
	if exists2 && solved1 {
		if b2 < -rc {
			solved2 = false
		} else {
			a0_new2 = ((a0-rc)%dim + dim) % dim
			diagR_map[0][a0_new2] = b2
			delete(diagR_map[2], a0)
			h_onetwo = append(h_onetwo, 2)
			h_a0 = append(h_a0, a0)
			h_a0_new = append(h_a0_new, a0_new2)
			h_b = append(h_b, b2)

			conflictNum_tmp := 0
			solved2, conflictNum_tmp = CheckZero2RCV2(diagR_map, a0_new2, b2, dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
			if solved2 {
				conflictNum = conflictNum + conflictNum_tmp + 1
			}
		}

		if !solved2 {
			solved = false
		}
	}
	if !solved {
		for i := range h_onetwo {
			delete(diagR_map[0], h_a0_new[i])
			diagR_map[h_onetwo[i]][h_a0[i]] = h_b[i]
		}
	}
	return solved, conflictNum
}

func CheckZero2RCV3(diagR_map []map[int]int, diagR_map1 []map[int]int, diagR_map2 []map[int]int, a0 int, b0 int, dim int, rc int) {
	b1, exists1 := diagR_map[1][a0]
	b2, exists2 := diagR_map[2][a0]
	var a0_new1 int
	var a0_new2 int

	if exists1 {
		if b1 > rc {
			_, exists := diagR_map1[1][a0]
			if !exists {
				diagR_map1[1][a0] = b1
				delete(diagR_map[1], a0)
			} else {
				diagR_map2[1][a0] = b1
				delete(diagR_map[1], a0)
			}
		} else {
			a0_new1 = ((a0+rc)%dim + dim) % dim
			diagR_map[0][a0_new1] = b1
			delete(diagR_map[1], a0)
			CheckZero2RCV3(diagR_map, diagR_map1, diagR_map2, a0_new1, b1, dim, rc)
		}

		/*
			a0_new1 = ((a0+rc)%dim + dim) % dim
			diagR_map[0][a0_new1] = b1
			delete(diagR_map[1], a0)
			h_onetwo = append(h_onetwo, 1)
			h_a0 = append(h_a0, a0)
			h_a0_new = append(h_a0_new, a0_new1)
			h_b = append(h_b, b1)

			solved1 = CheckZero2RCV2(diagR_map, a0_new1, b1, dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
			if !solved1 {
				solved = false

			}
		*/
	}

	if exists2 {
		if b2 < rc {
			/*
				_, exists := diagR_map2[2][a0]
				if !exists {
					diagR_map2[2][a0] = b2
					delete(diagR_map[2], a0)
				} else {
					diagR_map1[2][a0] = b2
					delete(diagR_map[2], a0)
				}
			*/
			_, exists := diagR_map1[2][a0]
			if !exists {
				diagR_map1[2][a0] = b2
				delete(diagR_map[2], a0)
			} else {
				diagR_map2[2][a0] = b2
				delete(diagR_map[2], a0)
			}
		} else {
			a0_new2 = ((a0-rc)%dim + dim) % dim
			diagR_map[0][a0_new2] = b2
			delete(diagR_map[2], a0)
			CheckZero2RCV3(diagR_map, diagR_map1, diagR_map2, a0_new2, b2, dim, rc)
		}
		/*
			a0_new2 = ((a0-rc)%dim + dim) % dim
			diagR_map[0][a0_new2] = b2
			delete(diagR_map[2], a0)
			h_onetwo = append(h_onetwo, 2)
			h_a0 = append(h_a0, a0)
			h_a0_new = append(h_a0_new, a0_new2)
			h_b = append(h_b, b2)

			solved2 = CheckZero2RCV2(diagR_map, a0_new2, b2, dim, rc, h_onetwo, h_a0, h_a0_new, h_b)
			if !solved2 {
				solved = false
			}
		*/
	}
	/*
		if !solved {
			for i := range h_onetwo {
				delete(diagR_map[0], h_a0_new[i])
				diagR_map[h_onetwo[i]][h_a0[i]] = h_b[i]
			}
		}
		return solved
	*/
}

func CentralizeKeys(diag map[int][]float64, d0 int, n int) map[int][]float64 {
	newDiag := make(map[int][]float64)
	for k := range diag {
		k_tmp := 0
		if k > n/2 {
			k_tmp = k - n
		} else {
			k_tmp = k
		}
		newDiag[k_tmp] = make([]float64, n)
		copy(newDiag[k_tmp], diag[k])
	}
	return newDiag

	/*
		keys := make([]int, 0, len(diag))
		for k := range diag {
			keys = append(keys, k)
		}
		sort.Ints(keys)

		mid := int(((keys[0]+keys[len(keys)-1])/d0)/2) * d0
		newDiag := make(map[int][]float64)
		for _, k := range keys {
			newKey := k - mid
			newDiag[newKey] = diag[k]
		}

		count_pos := 0
		count_neg := 0
		for key := range newDiag {
			if key > 0 {
				count_pos += 1
			} else if key < 0 {
				count_neg += 1
			}
		}

		if count_pos < count_neg {
			newDiag[keys[len(keys)-1]-mid+d0] = make([]float64, n)
		} else if count_neg < count_pos {
			newDiag[keys[0]-mid-d0] = make([]float64, n)
		}
		return newDiag
	*/
}

func ZeroDiagPadding(diags map[int][]float64, minKey int, maxKey int, n int) (newDiags map[int][]float64) {

	newDiags = make(map[int][]float64)

	for i := minKey; i <= maxKey; i++ {
		if v, ok := diags[i]; ok {
			newDiags[i] = v
		} else {
			newDiags[i] = make([]float64, n)
		}
	}
	return
}

func ZeroDiagPaddingV2(diags map[int][]float64, d0 int, n int) (newDiags map[int][]float64) {
	minKey := math.MaxInt
	maxKey := math.MinInt

	for key := range diags {
		if key < minKey {
			minKey = key
		}
		if key > maxKey {
			maxKey = key
		}
	}

	if (maxKey-minKey)%d0 != 0 {
		panic(errors.New("cannot find d0"))
	}

	newDiags = make(map[int][]float64)

	for i := minKey; i <= maxKey; i += d0 {
		if v, ok := diags[i]; ok {
			newDiags[i] = v
		} else {
			newDiags[i] = make([]float64, n)
		}
	}
	return
}

func Benes_network(n int, pi []int, min_n int) (rslt [][]int) {
	if (min_n) <= 2 {
		min_n = 2
	}
	if n <= min_n {
		rslt = make([][]int, 0)
		rslt = append(rslt, pi)
		return
	}

	piInv := make([]int, n)
	for idx, val := range pi {
		piInv[val] = idx
	}

	Lmap := make(map[int]int, 0)
	Rmap := make(map[int]int, 0)
	for i := 0; i < n; i++ {
		Lmap[i] = 1
		Rmap[i] = 1
	}

	circle_val := make([]int, 0)
	circle_flag := make([]string, 0)
	circle_color := make([]int, 0)

	start_flag := "L"
	start_node := 0
	start_color := 0
	start_edge := "p" // permutation

	flag := "L"
	current_node := 0
	color := 0
	edge := "p"
	for idx := range Lmap {
		start_flag = "L"
		start_node = idx
		start_color = 0
		start_edge = "p" // permutation

		flag = start_flag
		current_node = start_node
		color = start_color
		edge = start_edge
		for {
			circle_val = append(circle_val, current_node)
			circle_flag = append(circle_flag, flag)
			circle_color = append(circle_color, color)
			color = (color + 1) % 2
			if flag == "L" {
				delete(Lmap, current_node)
			} else {
				delete(Rmap, current_node)
			}

			if edge == "p" {
				if flag == "L" {
					flag = "R"
					current_node = pi[current_node]
				} else {
					flag = "L"
					current_node = piInv[current_node]
				}
				edge = "c"
			} else {
				if current_node < n/2 {
					current_node += n / 2
				} else {
					current_node -= n / 2
				}
				edge = "p"
			}

			if flag == start_flag && current_node == start_node {
				break
			}
		}
	}

	tau := make([]int, n)
	sigmaInv := make([]int, n)
	sigma := make([]int, n)

	for idx := range circle_val {
		i := circle_val[idx]
		if circle_flag[idx] == "L" {
			if (i < n/2) && (circle_color[idx] == 0) {
				tau[i] = i
			} else if (i >= n/2) && (circle_color[idx] == 1) {
				tau[i] = i
			} else if (i < n/2) && (circle_color[idx] == 1) {
				tau[i] = i + n/2
			} else if (i >= n/2) && (circle_color[idx] == 0) {
				tau[i] = i - n/2
			}
		} else {
			if (i < n/2) && (circle_color[idx] == 1) {
				sigmaInv[i] = i
			} else if (i >= n/2) && (circle_color[idx] == 0) {
				sigmaInv[i] = i
			} else if (i < n/2) && (circle_color[idx] == 0) {
				sigmaInv[i] = i + n/2
			} else if (i >= n/2) && (circle_color[idx] == 1) {
				sigmaInv[i] = i - n/2
			}
		}
	}

	for idx, val := range sigmaInv {
		sigma[val] = idx
	}

	rho := make([]int, n)

	for idx, val := range pi {
		input := tau[idx]
		output := sigmaInv[val]
		rho[input] = output
	}

	rho_dmp := make([][]int, 0)
	if n >= 2 {
		rslt_tophalf := Benes_network(n/2, rho[0:n/2], min_n)

		rho_bothalf_tmp := make([]int, n/2)
		for idx := range rho_bothalf_tmp {
			rho_bothalf_tmp[idx] = rho[idx+n/2] - n/2
		}
		rslt_bothalf := Benes_network(n/2, rho_bothalf_tmp, min_n)

		for idx := range rslt_tophalf {
			tmp := make([]int, n)
			for i, v := range rslt_tophalf[idx] {
				tmp[i] = v
			}
			for i, v := range rslt_bothalf[idx] {
				tmp[i+n/2] = v + n/2
			}
			rho_dmp = append(rho_dmp, tmp)
		}
	}

	rslt = make([][]int, 0)
	rslt = append(rslt, tau)
	rslt = append(rslt, rho_dmp...)
	rslt = append(rslt, sigma)

	return
}

func Mapping(vec []int, map_test []int) (rslt_vec []int) {
	rslt_vec = make([]int, len(vec))
	for idx, val := range map_test {
		rslt_vec[val] = vec[idx]
	}
	return
}

func Multi_mapping(vec []int, maps [][]int) (rslt_vec []int) {
	rslt_vec = make([]int, len(vec))
	for i := range rslt_vec {
		rslt_vec[i] = vec[i]
	}

	for _, val := range rslt_vec {
		fmt.Printf("%d ", val)
	}
	fmt.Printf("\n")
	for idx := len(maps) - 1; idx >= 0; idx-- {
		rslt_vec = Mapping(rslt_vec, maps[idx])
		for _, val := range rslt_vec {
			fmt.Printf("%d ", val)
		}
		fmt.Printf("\n")
	}
	return
}

func Permutation2Diagonals(map_test []int, n int) (diags map[int][]float64) {
	diags = make(map[int][]float64)
	for idx, val := range map_test {
		rot := ((idx - val) + n) % n
		_, exists := diags[rot]
		if !exists {
			diags[rot] = make([]float64, n)
		}
		diags[rot][val] = 1
	}
	return
}

func Multi_Permutation2Diagonals(maps [][]int, n int) (diags []map[int][]float64) {
	diags = make([]map[int][]float64, len(maps))
	for idx, val := range maps {
		diags[idx] = Permutation2Diagonals(val, n)
	}
	return
}

func GenerateRandomPermutation(n int) []int {
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}

	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	r.Shuffle(n, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})

	return perm
}

func GenerateRandomPermutationV2(n int, d0 int) []int {
	S := make(map[int]int)
	for i := 0; i < n; i++ {
		S[i] = 1
	}

	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	P := make([]map[int]int, 0)
	for i := range S {
		coset := make(map[int]int)
		coset[i] = i
		for j := (i + d0) % n; j != i; j = (j + d0) % n {
			coset[j] = j
		}

		coset_arr := make([]int, 0)
		coset_arr_permuted := make([]int, len(coset))

		for j := range coset {
			delete(S, j)
			coset_arr = append(coset_arr, j)
		}
		copy(coset_arr_permuted, coset_arr)
		r.Shuffle(len(coset_arr_permuted), func(i, j int) {
			coset_arr_permuted[i], coset_arr_permuted[j] = coset_arr_permuted[j], coset_arr_permuted[i]
		})
		for j := range coset_arr_permuted {
			coset[coset_arr[j]] = coset_arr_permuted[j]
		}
		P = append(P, coset)
	}
	perm := make([]int, n)
	for _, coset_map := range P {
		for i, j := range coset_map {
			perm[i] = j
		}
	}
	return perm
}

func GeneratePermutationWithDiag(n int, d0 int, b int) (diags map[int][]float64) {
	b_tmp := b
	for ; b_tmp*d0 >= n/2; b_tmp-- {

	}
	diags = make(map[int][]float64, 1+2*b_tmp)
	diags[0] = make([]float64, n)
	for i := 1; i <= b_tmp; i++ {
		diags[i*d0] = make([]float64, n)
		diags[-i*d0] = make([]float64, n)
	}

	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	cols := make([]int, n)

	for i := 0; i < len(cols); i++ {
		cols[i] = 0
	}
	for row := 0; row < n; row++ {
		diags_tmp := make(map[int][]float64)
		for key, val := range diags {
			diags_tmp[key] = make([]float64, n)
			copy(diags_tmp[key], val)
		}
		for len(diags_tmp) > 0 {
			keys := make([]int, len(diags_tmp))
			for k := range diags_tmp {
				keys = append(keys, k)
			}
			randomkey := keys[r.Intn(len(keys))]
			col := (row + randomkey + n) % n
			if cols[col] == 1 {
				delete(diags_tmp, randomkey)
			} else {
				cols[col] = 1
				diags[randomkey][row] = 1
				break
			}
		}
	}

	return
}

func Diagonals2Permutation(n int, diags map[int][]float64) (perms []int) {
	perms = make([]int, n)
	for rot, mask := range diags {
		for val, valval := range mask {
			if valval == 1.0 {
				perms[(rot+val+n)%n] = val
			}
		}
	}
	return
}

func ExtractNonzeroDiags(diags map[int][]float64) (new_diags map[int][]float64) {
	new_diags = make(map[int][]float64)
	for idx, arr := range diags {
		for _, val := range arr {
			if val != 0 {
				new_diags[idx] = make([]float64, len(arr))
				copy(new_diags[idx], arr)
				break
			}
		}
	}
	return
}

func ExtractNonzeroDiagsMulti(diags []map[int][]float64) (new_diags []map[int][]float64) {
	new_diags = make([]map[int][]float64, 0)
	for _, onemap := range diags {
		new_diags = append(new_diags, ExtractNonzeroDiags(onemap))
	}
	return
}

func LinTrans_Plain(vec []float64, mat [][]float64) (rslt []float64) {
	rslt = make([]float64, len(vec))
	for i := range rslt {
		rslt[i] = 0
		for j, entry := range mat[i] {
			rslt[i] += (entry * vec[j])
		}
	}
	return
}

func PermutationCollapse(perms [][]int) (new_perm []int) {
	new_perm = make([]int, len(perms[0]))
	for input := range new_perm {
		input_tmp := input
		for _, perm := range perms {
			input_tmp = perm[input_tmp]
		}
		new_perm[input] = input_tmp
	}
	return
}

func CostofCollapse(j1 int, j2 int, j_current int, n int) (cost_arr []int) {
	d := int(math.Log2(float64(n)))*2 - 1
	r := int(math.Log2(float64(n)))
	S_j := make([]int, 3)
	S_j[0] = 0
	if j_current < r {
		S_j[1] = (n / (1 << (j_current + 1)))
	} else {
		S_j[1] = (n / (1 << (d - j_current)))
	}
	S_j[2] = -S_j[1]

	cost_mat := make([][]int, 3)

	cost_map := make(map[int]bool)

	cost_arr = make([]int, 0)

	var cost_arr_tmp []int
	if j_current < j2 {
		cost_arr_tmp = CostofCollapse(j1, j2, j_current+1, n)
	}
	for idx, step := range S_j {
		if j_current < j2 {
			cost_mat[idx] = make([]int, len(cost_arr_tmp))
			copy(cost_mat[idx], cost_arr_tmp)
			for i := range cost_mat[idx] {
				cost_mat[idx][i] += step // shift cost
			}
		} else {
			cost_mat[idx] = make([]int, 1)
			cost_mat[idx][0] = step
		}
		for _, cost := range cost_mat[idx] {
			cost_reduced := (((cost % n) + n) % n) // reduced to rotation cost.
			if _, exists := cost_map[cost_reduced]; !exists {
				cost_map[cost_reduced] = true
				cost_arr = append(cost_arr, cost_reduced)
			}
		}
	}
	return
}

type Pair struct {
	J1 int
	J2 int
}

func OptCollapse(n int, B int, d_tmp int, B_tmp int, Cost_map map[Pair]int) (Cost int, pairofLayer [][]int) {
	d := 2*int(math.Log2(float64(n))) - 1
	if (B >= d) || (B_tmp >= d) {
		return 0, make([][]int, 0)
	} else if d_tmp == 0 {
		return 0, make([][]int, 0)
	} else if (d_tmp > 0) && (B_tmp == 0) {
		return math.MaxInt, make([][]int, 0)
	} else {
		Cost = math.MaxInt
		var pairJ1J2_opt [][]int
		for l := 1; l <= d_tmp; l++ {
			pairJ1J2 := Pair{J1: d_tmp - l, J2: d_tmp - 1}
			cost_tmp := 0
			cost_smaller, pairoflayer_tmp := OptCollapse(n, B, d_tmp-l, B_tmp-1, Cost_map)
			if _, exists := Cost_map[pairJ1J2]; !exists {
				Cost_map[pairJ1J2] = len(CostofCollapse(d_tmp-l, d_tmp-1, d_tmp-l, n))

			}
			if cost_smaller < math.MaxInt {
				cost_tmp = Cost_map[pairJ1J2] + cost_smaller
			} else {
				cost_tmp = math.MaxInt
			}
			if cost_tmp < Cost {
				Cost = cost_tmp
				pairJ1J2_opt = make([][]int, len(pairoflayer_tmp))
				copy(pairJ1J2_opt, pairoflayer_tmp)
				single_pair_arr := make([]int, 2)
				single_pair_arr[0] = pairJ1J2.J1
				single_pair_arr[1] = pairJ1J2.J2
				pairJ1J2_opt = append(pairJ1J2_opt, single_pair_arr)
			}

		}
		return Cost, pairJ1J2_opt
	}

}

var _aa = 0
var _ac = 1
var _ca = 2
var _cc = 3
var _cb = 4
var _bc = 5
var _bb = 6

var _ab = 7
var _ba = 8

var _a = 0
var _c = 1
var _b = 2

func maxRotAbs(diag []int) (max int) {
	max = diag[0]
	min := diag[0]
	for i := 1; i < len(diag); i++ {
		if diag[i] > max {
			max = diag[i]
		}
		if diag[i] < min {
			min = diag[i]
		}
	}
	if min < 0 {
		min = -min
	}
	if max < 0 {
		max = -max
	}
	if min > max {
		return min
	}
	return max
}

func minRotAbs(diag []int) (min int) {
	min = 32768
	for i := 1; i < len(diag); i++ {
		if diag[i] > 0 && min > diag[i] {
			min = diag[i]
		} else if diag[i] < 0 && min > -diag[i] {
			min = -diag[i]
		}
	}
	return min
}

func getMargin(diag []int) int {
	rotmap := make(map[int]bool)
	for i := 0; i < len(diag); i++ {
		if diag[i] == 0 {
			continue
		}
		rotmap[diag[i]] = true
	}
	return 0
}

func Depair(ab string) (a int, b int) {
	fmt.Sscanf(ab, "%d:%d", &a, &b)
	return
}

func InversePair(ab string) (ba string) {
	a, b := Depair(ab)
	ba = fmt.Sprintf("%d:%d", b, a)
	return
}

func rotate(a []int, r int) []int {
	b := make([]int, len(a))
	for i := 0; i < len(a); i++ {
		b[(len(a)+((i-r)%len(a)))%len(a)] = a[i]
	}
	return b
}

func mul(a []int, b []int) []int {
	c := make([]int, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] * b[i]
	}
	return c
}

func Add(a []int, b []int) []int {
	if a == nil {
		return b
	}
	c := make([]int, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] + b[i]
	}
	return c
}

func Add2(a []int, b []int) []int {
	if a == nil {
		tmp := make([]int, len(b))
		copy(tmp, b)
		return tmp
	}
	c := make([]int, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] + b[i]
	}
	return c
}

func pair(a, b int) string {
	return strconv.Itoa(a) + ":" + strconv.Itoa(b)
}

func set(musk map[string][]int, pair string, recv int, len int) {
	if musk[pair] == nil {
		musk[pair] = make([]int, len)
		for i := 0; i < len; i++ {
			musk[pair][i] = 0
		}
	}
	musk[pair][recv] = 1
}

func unset(musk map[string][]int, pair string, recv int, len int) {
	if musk[pair] == nil {
		panic(errors.New("musk[pair]==nil, cannot unset"))
	} else {
		if musk[pair][recv] == 0 {
			fmt.Printf("musk[%s][%d] is zero already\n", pair, recv)
		} else {
			musk[pair][recv] = 0
		}
	}
}

func ClearZeroMask(mask []int) []int {

	for _, val := range mask {
		if val != 0 {
			return mask
		}
	}

	return nil

}

func DecodeMap(_xy int) (x int, y int) {
	if _xy == _aa {
		return _a, _a
	} else if _xy == _ac {
		return _a, _c
	} else if _xy == _ca {
		return _c, _a
	} else if _xy == _cc {
		return _c, _c
	} else {
		return -1, -1
	}
}

func check(visited *[][]int, recv int) int {
	for i := 0; i < len(*visited); i++ {
		if (*visited)[i][recv] == 0 {
			(*visited)[i][recv] = 1
			return i
		}
	}
	*visited = append(*visited, make([]int, len(((*visited)[0]))))
	for i := 0; i < len(*visited); i++ {
		(*visited)[len(*visited)-1][i] = 0
	}
	(*visited)[len(*visited)-1][recv] = 1
	return len(*visited) - 1
}

func Compare(a, b []int) bool {

	rslt := true
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			rslt = false
			fmt.Printf("Diff a[%d]-b[%d]=%d\n", i, i, a[i]-b[i])
		}
	}
	fmt.Printf("diff: [")
	for i := range a {
		fmt.Printf("%d, ", a[i]-b[i])
	}
	fmt.Printf("]\n")
	return rslt
}

func ApproxCompare(a, b []float64, deci_prec float64) bool {
	if len(a) != len(b) {
		fmt.Printf("len(a):%d != len(b): %d\n", len(a), len(b))
		return false
	}
	for i := range a {
		diff := math.Abs(a[i] - b[i])
		if diff > deci_prec {
			fmt.Printf("diff %f > prec at index %d\n", diff, i)
			fmt.Printf("[")
			for i := 0; i < len(a); i++ {
				fmt.Printf("%.0f, ", a[i])
			}
			fmt.Printf("]\n")
			fmt.Printf("\n")
			fmt.Printf("[")
			for i := 0; i < len(b); i++ {
				fmt.Printf("%.0f, ", b[i])
			}
			fmt.Printf("]\n")
			fmt.Printf("\n")

			return false
		}
	}
	return true
}

func Transform(num []int, depth int, rot []int, musk [][]map[string][]int) []int {
	pre := make([]map[int][]int, 3)
	for i := 0; i < 3; i++ {
		pre[i] = make(map[int][]int)
	}
	pre[_c][0] = num

	for i := depth; i >= 0; i-- { // for i := depth; i > 0; i--
		for k, _ := range pre[_a] {
			pre[_a][k] = rotate(pre[_a][k], rot[i])
		}
		for k, _ := range pre[_b] {
			pre[_b][k] = rotate(pre[_b][k], -rot[i])
		}

		nex := make([]map[int][]int, 3)
		for j := 0; j < 3; j++ {
			nex[j] = make(map[int][]int)
		}

		for k, v := range musk[i][_aa] {
			s, t := Depair(k)
			if len(pre[_a][s]) == 0 {
				fmt.Printf("GGG")
			}

			nex[_a][t] = Add(nex[_a][t], mul(pre[_a][s], v))
			if len(nex[_a][t]) == 0 {
				fmt.Printf("fff")
			}

		}
		for k, v := range musk[i][_ac] {
			s, t := Depair(k)

			if len(pre[_a][s]) == 0 {
				fmt.Printf("GGG")
			}
			nex[_c][t] = Add(nex[_c][t], mul(pre[_a][s], v))
			if len(nex[_c][t]) == 0 {
				fmt.Printf("fff")
			}
		}
		for k, v := range musk[i][_ca] {
			s, t := Depair(k)
			if len(pre[_c][s]) == 0 {
				fmt.Printf("GGG")
			}
			nex[_a][t] = Add(nex[_a][t], mul(pre[_c][s], v))
			if len(nex[_a][t]) == 0 {
				fmt.Printf("fff")
			}
		}
		for k, v := range musk[i][_cc] {
			s, t := Depair(k)
			if len(pre[_c][s]) == 0 {
				fmt.Printf("GGG")
			}
			nex[_c][t] = Add(nex[_c][t], mul(pre[_c][s], v))
			if len(nex[_c][t]) == 0 {
				fmt.Printf("fff")
			}
		}
		for k, v := range musk[i][_cb] {
			s, t := Depair(k)
			if len(pre[_c][s]) == 0 {
				fmt.Printf("GGG")
			}
			nex[_b][t] = Add(nex[_b][t], mul(pre[_c][s], v))
			if len(nex[_b][t]) == 0 {
				fmt.Printf("fff")
			}
		}
		for k, v := range musk[i][_bc] {
			s, t := Depair(k)
			if len(pre[_b][s]) == 0 {
				fmt.Printf("GGG")
			}
			nex[_c][t] = Add(nex[_c][t], mul(pre[_b][s], v))
			if len(nex[_c][t]) == 0 {
				fmt.Printf("fff")
			}
		}
		for k, v := range musk[i][_bb] {
			s, t := Depair(k)
			if len(pre[_b][s]) == 0 {
				fmt.Printf("GGG")
			}
			nex[_b][t] = Add(nex[_b][t], mul(pre[_b][s], v))
			if len(nex[_b][t]) == 0 {
				fmt.Printf("fff")
			}
		}

		pre = nex
	}
	return pre[_c][0]
}

func TransformV2(num []int, RRot []map[int]int, MMask [][][][]map[string][]int) []int {

	pre := make([][]map[int][]int, len(RRot))
	for i := range pre {
		pre[i] = make([]map[int][]int, 2)
		for j := range pre[i] {
			pre[i][j] = make(map[int][]int)
		}
		pre[i][_c][0] = make([]int, len(num))
		copy(pre[i][_c][0], num)
	}

	lv_max := 0
	for m := range RRot {
		if lv_max < len(RRot[m])-1 {
			lv_max = len(RRot[m]) - 1
		}
	}

	for lv := lv_max; lv >= 0; lv-- {

		nex := make([][]map[int][]int, len(RRot))
		for i := range nex {
			nex[i] = make([]map[int][]int, 2)
			for j := range nex[i] {
				nex[i][j] = make(map[int][]int)
			}
		}
		nex_visited := make(map[int]bool, 0)
		for i := range nex {
			nex_visited[i] = false
		}

		for m := range RRot {
			for k, _ := range pre[m][_a] {
				if _, exists := RRot[m][lv]; exists {
					pre[m][_a][k] = rotate(pre[m][_a][k], RRot[m][lv])
				}
			}

			for i_g := range MMask[m] {
				if len(MMask[m][i_g]) <= lv || MMask[m][i_g][lv] == nil {
					continue
				}
				if MMask[m][i_g][lv][_aa] != nil {
					for k, v := range MMask[m][i_g][lv][_aa] {
						s, t := Depair(k)
						if len(pre[m][_a][s]) == 0 {
							panicinfo := fmt.Sprintf("len(pre[%d][%d][%d])==0", m, _a, s)
							panic(errors.New(panicinfo))
						}
						nex[i_g][_a][t] = Add2(nex[i_g][_a][t], mul(pre[m][_a][s], v))
						nex_visited[i_g] = true
					}
				}
				if MMask[m][i_g][lv][_ac] != nil {
					for k, v := range MMask[m][i_g][lv][_ac] {
						s, t := Depair(k)
						if len(pre[m][_a][s]) == 0 {
							panicinfo := fmt.Sprintf("len(pre[%d][%d][%d])==0", m, _a, s)
							panic(errors.New(panicinfo))
						}
						nex[i_g][_c][t] = Add2(nex[i_g][_c][t], mul(pre[m][_a][s], v))
						nex_visited[i_g] = true
					}
				}
				if MMask[m][i_g][lv][_ca] != nil {
					for k, v := range MMask[m][i_g][lv][_ca] {
						s, t := Depair(k)
						if len(pre[m][_c][s]) == 0 {
							panicinfo := fmt.Sprintf("len(pre[%d][%d][%d])==0", m, _c, s)
							panic(errors.New(panicinfo))
						}
						nex[i_g][_a][t] = Add2(nex[i_g][_a][t], mul(pre[m][_c][s], v))
						nex_visited[i_g] = true
					}
				}
				if MMask[m][i_g][lv][_cc] != nil {
					for k, v := range MMask[m][i_g][lv][_cc] {
						s, t := Depair(k)
						if len(pre[m][_c][s]) == 0 {
							panicinfo := fmt.Sprintf("len(pre[%d][%d][%d])==0", m, _c, s)
							panic(errors.New(panicinfo))
						}
						nex[i_g][_c][t] = Add2(nex[i_g][_c][t], mul(pre[m][_c][s], v))
						nex_visited[i_g] = true
					}
				}

			}

		}

		/*
			for m := range RRot {
				if _, exists := RRot[m][lv]; exists {
					if (RRot[m][lv] != 0) || (RRot[m][lv] == 0 && lv == lv_max) {
						pre[m] = nex[m]
					}
				}
			}
		*/
		for m := range nex {
			if nex_visited[m] {
				pre[m] = nex[m]
			}
		}

		for i_g := range nex {
			for s := range nex[i_g] {
				for t := range nex[i_g][s] {
					fmt.Printf("group %d, level %d nex[%d][%d]:\n", i_g, lv, s, t)
					auxiliary_io.Print_vector_int_full(nex[i_g][s][t])
					fmt.Printf("\n")
				}
			}
		}

	}

	/*
		rslt := make([]int, len(num))

		for m := range pre {
			rslt = Add(rslt, pre[m][_c][0])
		}
	*/

	return pre[0][_c][0]

}

func TransformV3(num []int, RRot []map[int]int, MMask [][][][]map[string][]int, inv bool) []int {
	pre := make([][]map[int][]int, len(RRot))
	for i := range pre {
		pre[i] = make([]map[int][]int, 2)
		for j := range pre[i] {
			pre[i][j] = make(map[int][]int)
		}
		if !inv {
			pre[i][_c][0] = make([]int, len(num))
			copy(pre[i][_c][0], num)
		}
	}
	pre[0][_c][0] = make([]int, len(num))
	copy(pre[0][_c][0], num)

	lv_max := 0
	for m := range RRot {
		if lv_max < len(RRot[m])-1 {
			lv_max = len(RRot[m]) - 1
		}
	}

	var lv_begin int
	var lv_end int
	var inc int
	var MMask_tmp [][][][]map[string][]int
	var RRot_tmp []map[int]int
	if !inv {
		lv_begin = lv_max
		lv_end = 0
		inc = -1
	} else {
		lv_begin = 0
		lv_end = lv_max
		inc = 1
		MMask_tmp = make([][][][]map[string][]int, len(MMask))
		RRot_tmp = make([]map[int]int, len(RRot))
		for m := range RRot {
			RRot_tmp[m] = make(map[int]int)
			for k := range RRot[m] {
				if k < lv_max {
					RRot_tmp[m][k+1] = -RRot[m][k]
				}
			}
			RRot_tmp[m][0] = 0
		}
		RRot = RRot_tmp

		for m := range MMask {

			for i_g := range MMask[m] {
				if len(MMask_tmp[i_g])-1 < m {
					MMask_tmp[i_g] = append(MMask_tmp[i_g], make([][][]map[string][]int, m-len(MMask_tmp[i_g])+1)...)
				}
				for lv := range MMask[m][i_g] {
					for cls := range MMask[m][i_g][lv] {

						if len(MMask_tmp[i_g][m])-1 < lv {
							MMask_tmp[i_g][m] = append(MMask_tmp[i_g][m], make([][]map[string][]int, lv-len(MMask_tmp[i_g][m])+1)...)
						}

						var cls_inv int
						if cls == _ac {
							cls_inv = _ca
						} else if cls == _ca {
							cls_inv = _ac
						} else {
							cls_inv = cls
						}

						if len(MMask_tmp[i_g][m][lv])-1 < cls_inv {
							MMask_tmp[i_g][m][lv] = append(MMask_tmp[i_g][m][lv], make([]map[string][]int, cls_inv-len(MMask_tmp[i_g][m][lv])+1)...)
						}
						if MMask_tmp[i_g][m][lv][cls_inv] == nil {
							MMask_tmp[i_g][m][lv][cls_inv] = make(map[string][]int)
						}

						for k, v := range MMask[m][i_g][lv][cls] {
							k_inv := InversePair(k)
							MMask_tmp[i_g][m][lv][cls_inv][k_inv] = make([]int, len(v))
							copy(MMask_tmp[i_g][m][lv][cls_inv][k_inv], v)
						}

					}
				}
			}
		}
		MMask = MMask_tmp
	}

	for lv := lv_begin; ((lv >= lv_end) && (!inv)) || ((lv <= lv_end) && (inv)); lv = lv + inc {

		nex := make([][]map[int][]int, len(RRot))
		for i := range nex {
			nex[i] = make([]map[int][]int, 2)
			for j := range nex[i] {
				nex[i][j] = make(map[int][]int)
			}
		}

		nex_visited := make(map[int]bool, 0)
		for i := range nex {
			nex_visited[i] = false
		}

		for m := range RRot {
			for k, _ := range pre[m][_a] {
				if _, exists := RRot[m][lv]; exists {
					pre[m][_a][k] = rotate(pre[m][_a][k], RRot[m][lv])
				}
			}

			for i_g := range MMask[m] {
				if len(MMask[m][i_g]) <= lv || MMask[m][i_g][lv] == nil {
					continue
				}
				if MMask[m][i_g][lv][_aa] != nil {
					for k, v := range MMask[m][i_g][lv][_aa] {
						s, t := Depair(k)
						if len(pre[m][_a][s]) == 0 {
							panicinfo := fmt.Sprintf("len(pre[%d][%d][%d])==0", m, _a, s)
							panic(errors.New(panicinfo))
						}
						nex[i_g][_a][t] = Add2(nex[i_g][_a][t], mul(pre[m][_a][s], v))
						nex_visited[i_g] = true
					}
				}
				if MMask[m][i_g][lv][_ac] != nil {
					for k, v := range MMask[m][i_g][lv][_ac] {
						s, t := Depair(k)
						if len(pre[m][_a][s]) == 0 {
							panicinfo := fmt.Sprintf("len(pre[%d][%d][%d])==0", m, _a, s)
							panic(errors.New(panicinfo))
						}
						nex[i_g][_c][t] = Add2(nex[i_g][_c][t], mul(pre[m][_a][s], v))
						nex_visited[i_g] = true
					}
				}
				if MMask[m][i_g][lv][_ca] != nil {
					for k, v := range MMask[m][i_g][lv][_ca] {
						s, t := Depair(k)
						if len(pre[m][_c][s]) == 0 {
							panicinfo := fmt.Sprintf("len(pre[%d][%d][%d])==0", m, _c, s)
							panic(errors.New(panicinfo))
						}
						nex[i_g][_a][t] = Add2(nex[i_g][_a][t], mul(pre[m][_c][s], v))
						nex_visited[i_g] = true
					}
				}
				if MMask[m][i_g][lv][_cc] != nil {
					for k, v := range MMask[m][i_g][lv][_cc] {
						s, t := Depair(k)
						if len(pre[m][_c][s]) == 0 {
							panicinfo := fmt.Sprintf("len(pre[%d][%d][%d])==0", m, _c, s)
							panic(errors.New(panicinfo))
						}
						nex[i_g][_c][t] = Add2(nex[i_g][_c][t], mul(pre[m][_c][s], v))
						nex_visited[i_g] = true
					}
				}

			}

		}

		/*
			for m := range RRot {
				if _, exists := RRot[m][lv]; exists {
					if (RRot[m][lv] != 0) || (RRot[m][lv] == 0 && lv == lv_begin) {
						pre[m] = nex[m]
					}
				}
			}
		*/
		for m := range nex {
			if nex_visited[m] {
				pre[m] = nex[m]
			}
		}

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
						fmt.Printf("group %d, level %d nex[%d][%d]\n", i_g, lv, s, t)
						for i := range nex[i_g][s][t] {
							if nex[i_g][s][t][i] == 7076 {
								fmt.Printf("has 7076 at pos %d\n", i)
							}
						}
					}
				}
			}
		*/
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
						fmt.Printf("group %d, level %d nex[%d][%d]:\n", i_g, lv, s, t)
						auxiliary_io.Print_vector_int_full(nex[i_g][s][t])
						fmt.Printf("\n")
					}
				}
			}
		*/
	}

	if !inv {
		return pre[0][_c][0]
	} else {
		rslt := make([]int, len(num))

		for m := range pre {
			rslt = Add(rslt, pre[m][_c][0])
		}
		return rslt
	}

}

func Permutation2DiagnalV2(permutation []int) [][]int {
	diag := make([][]int, 3)
	diag[0] = make([]int, len(permutation))
	for i := 0; i < len(permutation); i++ {
		diag[0][permutation[i]] = permutation[i] - i

		/*
			if diag[0][permutation[i]] > len(permutation)/2 {
				diag[0][permutation[i]] = diag[0][permutation[i]] - len(permutation)
			} else if diag[0][permutation[i]] < -len(permutation)/2 {
				diag[0][permutation[i]] = diag[0][permutation[i]] + len(permutation)
			}
		*/

		if diag[0][permutation[i]] < 0 {
			diag[0][permutation[i]] = diag[0][permutation[i]] + len(permutation)
		}
	}
	diag[1] = make([]int, len(permutation))
	for i := 0; i < len(permutation); i++ {
		diag[1][i] = 0
	}
	diag[2] = make([]int, len(permutation))
	for i := 0; i < len(permutation); i++ {
		diag[2][i] = _c
	}
	return diag
}

func SplitConflicts(diag1 [][]int, done map[int]int, binary bool) (diag2 [][]int, count int, marks map[int]int, done_new map[int]int) {
	diag := make([][]int, len(diag1))
	for i := 0; i < len(diag1); i++ {
		diag[i] = make([]int, len(diag1[0]))
		copy(diag[i], diag1[i])
	}

	diag2 = make([][]int, len(diag1))
	for i := 0; i < len(diag1); i++ {
		diag2[i] = make([]int, len(diag1[0]))
	}

	marks = make(map[int]int, 0)
	done_new = make(map[int]int, 0)

	count = 0
	min := minRotAbs(diag[0])
	max := maxRotAbs(diag[0])
	getMargin(diag[0])
	fmt.Println("min", min, " max", max)

	var rot int
	var rem int
	if binary {
		if max > 0 {
			maxlen := bits.Len(uint(max))
			if (1 << maxlen) >= max {
				rot = (1 << (maxlen - 1))
			} else {
				rot = (1 << maxlen)
			}
			rem = rot - 1
		} else {
			rot = 0
			rem = 0
		}

	} else {
		rem = max / 2
		rot = max - rem
	}

	for max := maxRotAbs(diag[0]); max > 0; max = maxRotAbs(diag[0]) {
		if binary {
			if max > 0 {
				maxlen := bits.Len(uint(max))
				if (1 << maxlen) >= max {
					rot = (1 << (maxlen - 1))
				} else {
					rot = (1 << maxlen)
				}
				rem = rot - 1
			} else {
				rot = 0
				rem = 0
			}

		} else {
			rem = max / 2
			rot = max - rem
		}

		visited := make([][][]int, 3)
		for i := 0; i < 3; i++ {
			visited[i] = make([][]int, 1)
			visited[i][0] = make([]int, len(diag[0]))
			for j := 0; j < len(diag[0]); j++ {
				visited[i][0][j] = 0
			}
		}

		flag := 0

	conflict:
		for i := 0; i < len(diag[0]); i++ {

			if _, exists := done[i]; exists {
				continue
			}

			if diag[0][i] > rem {

				if flag == 0 && diag[2][i] != _a {
					continue
				}

				newDiag := diag[0][i] - rot
				recv := (len(diag[0]) + i - newDiag) % len(diag[0])

				if visited[_a][0][recv] == 1 { // restricting type _a only contain one node (i.e. only one rotation node)
					newDiag = 0

					diag2[0][i] = diag1[0][i]
					diag2[1][i] = diag1[1][i]
					diag2[2][i] = diag1[2][i]

					diag1[0][i] = 0 // conflicting point moved to next group is regarded as complete in this group.
					marks[i] = 1

					count++
				} else {
					visited[_a][0][recv] = 1

				}

				diag[0][i] = newDiag
				diag[2][i] = _a
			} else if diag[0][i] < -rem {

				if flag == 0 && diag[2][i] != _b {
					continue
				}

				newDiag := diag[0][i] + rot
				recv := (len(diag[0]) + i - newDiag) % len(diag[0])

				if visited[_b][0][recv] == 1 { // restricting type _a only contain one node (i.e. only one rotation node)
					newDiag = 0

					diag2[0][i] = diag1[0][i]
					diag2[1][i] = diag1[1][i]
					diag2[2][i] = diag1[2][i]

					diag1[0][i] = 0 // conflicting point moved to next group is regarded as complete in this group.
					marks[i] = 1

					count++
				} else {
					visited[_b][0][recv] = 1

				}

				diag[0][i] = newDiag
				diag[2][i] = _b
			} else if flag == 0 && diag[0][i] != 0 {
				diag[2][i] = _c

			}
		}
		if flag == 0 {
			flag = 1
			goto conflict
		}
	}

	for i := 0; i < len(diag[0]); i++ {
		done_new[i] = 1
	}
	for i := range marks {
		delete(done_new, i)
	}

	fmt.Println(count)
	return diag2, count, marks, done_new
}

func DecomposeV3(diag [][]int, marks map[int]int, done map[int]int, binary bool) (musk []map[string][]int, rot int, visited [][][]int) {

	diag_backup := make([]map[int]int, 3)
	for i := 0; i < 3; i++ {
		diag_backup[i] = make(map[int]int)
	}

	max := maxRotAbs(diag[0])

	var rem int
	if binary {
		if max > 0 {
			maxlen := bits.Len(uint(max))
			if (1 << maxlen) >= max {
				rot = (1 << (maxlen - 1))
			} else {
				rot = (1 << maxlen)
			}
			rem = rot - 1
		} else {
			rot = 0
			rem = 0
		}

	} else {
		rem = max / 2
		rot = max - rem
	}

	visited = make([][][]int, 3)
	for i := 0; i < 3; i++ {
		visited[i] = make([][]int, 1)
		visited[i][0] = make([]int, len(diag[0]))
		for j := 0; j < len(diag[0]); j++ {
			visited[i][0][j] = 0
		}
	}

	musk = make([]map[string][]int, 9)
	for i := 0; i < 9; i++ {
		musk[i] = make(map[string][]int)
	}

	traversed := make(map[int]int)

	flag := 0
conflict:
	for i := 0; i < len(diag[0]); i++ {
		if _, exists := marks[i]; exists {
			continue
		}

		if _, exists := done[i]; exists {
			continue
		}

		if diag[0][i] > rem {

			if diag[2][i] != _a && flag == 0 {
				continue
			}
			/*
				if diag[2][i] == _a && flag == 1 {
					continue
				}
			*/

			if _, exists := traversed[i]; exists {
				continue //fmt.Printf("!!!")
			} else {
				traversed[i] = 1
			}

			newDiag := diag[0][i] - rot
			recv := (len(diag[0]) + i - newDiag) % len(diag[0])

			var pos int
			pos = check(&visited[_a], recv)

			/*
				if diag[2][i] != _a {
					pos = check(&visited[_a], recv)
				} else {
					if diag[1][i] >= len(visited[_a]) {
						tmp := make([][]int, diag[1][i]+1)
						copy(tmp, visited[_a])
						for i := len(visited[_a]); i < len(tmp); i++ {
							tmp[i] = make([]int, len(visited[_a][0]))
						}
						visited[_a] = tmp
					}
					if visited[_a][diag[1][i]][recv] == 1 {
						fmt.Printf("???")
					}
					visited[_a][diag[1][i]][recv] = 1
					pos = diag[1][i]
					diag_backup[0][i] = diag[0][i]
					diag_backup[1][i] = diag[1][i]
					diag_backup[2][i] = diag[2][i]
				}
			*/

			switch diag[2][i] {
			case _a:
				set(musk[_aa], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			case _c:
				set(musk[_ac], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			}

			diag[0][i] = newDiag
			diag[1][i] = pos
			diag[2][i] = _a
		} else if diag[0][i] < -rem {
			if diag[2][i] != _b && flag == 0 {
				continue
			}
			if diag[2][i] == _b && flag == 1 {
				continue
			}

			if _, exists := traversed[i]; exists {
				continue // fmt.Printf("!!!")
			} else {
				traversed[i] = 1
			}

			newDiag := diag[0][i] + rot
			recv := (len(diag[0]) + i - newDiag) % len(diag[0])

			var pos int

			if diag[2][i] != _b {
				pos = check(&visited[_b], recv)
			} else {
				if diag[1][i] >= len(visited[_b]) {
					tmp := make([][]int, diag[1][i]+1)
					copy(tmp, visited[_b])
					for i := len(visited[_b]); i < len(tmp); i++ {
						tmp[i] = make([]int, len(visited[_b][0]))
					}
					visited[_b] = tmp
				}
				if visited[_b][diag[1][i]][recv] == 1 {
					fmt.Printf("???")
				}
				visited[_b][diag[1][i]][recv] = 1
				pos = diag[1][i]
				diag_backup[0][i] = diag[0][i]
				diag_backup[1][i] = diag[1][i]
				diag_backup[2][i] = diag[2][i]
			}

			switch diag[2][i] {
			case _b:
				set(musk[_bb], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			case _c:
				set(musk[_bc], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			}

			diag[0][i] = newDiag
			diag[1][i] = pos
			diag[2][i] = _b

		} else {

			if diag[2][i] != _c && flag == 0 {
				continue
			}
			if diag[2][i] == _c && flag == 1 {
				continue
			}

			if _, exists := traversed[i]; exists {
				continue
			} else {
				traversed[i] = 1
			}

			newDiag := diag[0][i] - 0
			recv := (len(diag[0]) + i - newDiag) % len(diag[0])

			var pos int
			if diag[2][i] != _c {
				pos = check(&visited[_c], recv)
			} else {

				if rot > 0 {

					if diag[1][i] >= len(visited[_c]) {
						tmp := make([][]int, diag[1][i]+1)
						copy(tmp, visited[_c])
						for i := len(visited[_c]); i < len(tmp); i++ {
							tmp[i] = make([]int, len(visited[_c][0]))
						}
						visited[_c] = tmp
					}
					if visited[_c][diag[1][i]][recv] == 1 {
						fmt.Printf("???")
					}
					visited[_c][diag[1][i]][recv] = 1
					pos = diag[1][i]
					diag_backup[0][i] = diag[0][i]
					diag_backup[1][i] = diag[1][i]
					diag_backup[2][i] = diag[2][i]
				} else {
					pos = check(&visited[_c], recv)
				}

			}

			switch diag[2][i] {
			case _a:
				set(musk[_ca], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			case _b:
				set(musk[_cb], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			case _c:
				set(musk[_cc], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			}

			diag[1][i] = pos
			diag[2][i] = _c
		}
	}
	if flag == 0 {

		/*
			merge_map := make([]map[int]int, 3)

			for i := range visited {

				merge_map[i] = make(map[int]int)
				for j := 0; j < len(visited[i]); j++ {
					merge_map[i][j] = j
				}

				for j := len(visited[i]) - 1; j > 0; j-- { // do not traverse the first ciphertext.
					can_merge := true
					for k := range visited[i][j] {
						if visited[i][j-1][k] == 1 && visited[i][j][k] == 1 {
							can_merge = false
						}
					}
					if can_merge {

						merge_map[i][j] = j - 1 // store the relation between old index and the new one.
						for j_r := range merge_map[i] {
							if j_r > j {
								j_l := merge_map[i][j_r]
								merge_map[i][j_r] = j_l - 1
							}
						}

					}
				}

				var idx int
				if i == _a {
					idx = _aa
				} else if i == _b {
					idx = _bb
				} else {
					idx = _cc
				}

				for j := 0; j < len(visited[i]); j++ {
					j_l := merge_map[i][j]
					if j_l < j {
						if visited[i][j] == nil {
							visited[i][j] = make([]int, len(visited[i][0]))
						}
						if visited[i][j_l] == nil {
							visited[i][j_l] = make([]int, len(visited[i][0]))
						}
						for k := range visited[i][j] {
							if visited[i][j][k] == 1 {
								visited[i][j_l][k] = 1

							}
						}
						visited[i][j] = nil
					}

				}
				visited_tmp := make([][]int, 0)
				for j := range visited[i] {
					if visited[i][j] != nil {
						visited_tmp = append(visited_tmp, visited[i][j])
						merge_map[i][j] = len(visited_tmp) - 1
						for j_tmp := range merge_map[i] {
							if merge_map[i][j_tmp] == j {
								merge_map[i][j_tmp] = len(visited_tmp) - 1
							}
						}
					}
				}
				visited[i] = visited_tmp

				for j, j_l := range merge_map[i] {
					if j_l < j {

						for recv, val := range musk[idx][pair(j, j)] {
							if val == 1 {
								set(musk[idx], pair(j_l, j), recv, len(diag[0]))
							}
						}
						delete(musk[idx], pair(j, j))
					}
				}

			}
			for i := range diag_backup[0] {
				if diag_backup[0][i] > rem {
					diag[1][i] = merge_map[_a][diag[1][i]]
				} else if diag_backup[0][i] < -rem {
					diag[1][i] = merge_map[_b][diag[1][i]]
				} else {
					diag[1][i] = merge_map[_c][diag[1][i]]
				}
			}
		*/
		flag = 1
		goto conflict
	}
	return

}

func ReduceMasks(mask [][]map[string][]int) (mask_reduced [][]map[string][]int) {
	mask_reduced = make([][]map[string][]int, len(mask))
	depth := len(mask) - 1

	count := 0
	for i := depth; i >= 0; i-- {
		for j := 0; j < len(mask[i]); j++ {
			if mask[i][j] != nil {
				count += len(mask[i][j])
			}
		}
	}
	fmt.Printf("Original Mask Number: %d\n", count)

	return

}

func DecomposeV4(diag [][]int, marks map[int]int, done map[int]int) (musk []map[string][]int, rot int, visited [][][]int) {
	max := maxRotAbs(diag[0])
	rem := max / 2
	rot = max - rem

	visited = make([][][]int, 3)
	for i := 0; i < 3; i++ {
		visited[i] = make([][]int, 1)
		visited[i][0] = make([]int, len(diag[0]))
		for j := 0; j < len(diag[0]); j++ {
			visited[i][0][j] = 0
		}
	}

	musk = make([]map[string][]int, 9)
	for i := 0; i < 9; i++ {
		musk[i] = make(map[string][]int)
	}

	traversed := make(map[int]int)

	diag_ca := make(map[int][][]int) // record _ca's recv that has already been handled.

	for i := 0; i < len(diag[0]); i++ {

		if _, exists := marks[i]; exists {
			continue
		}

		if _, exists := done[i]; exists {
			continue
		}

		if diag[0][i] > rem {

			traversed[i] = 1

			newDiag := diag[0][i] - rot
			recv := (len(diag[0]) + i - newDiag) % len(diag[0])
			pos := check(&visited[_a], recv)

			switch diag[2][i] {
			case _a:
				set(musk[_aa], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			case _c:
				set(musk[_ac], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			}
			diag[0][i] = newDiag
			diag[1][i] = pos
			diag[2][i] = _a
		} else if (diag[0][i] <= rem) && (diag[0][i] >= -rem) && (diag[2][i] == _a) {

			traversed[i] = 1

			newDiag := diag[0][i] - 0
			recv := (len(diag[0]) + i - newDiag) % len(diag[0])
			pos := check(&visited[_c], recv)

			set(musk[_ca], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))

			diag[1][i] = pos
			diag[2][i] = _c

			if _, exists := diag_ca[pos]; !exists {
				diag_ca[pos] = make([][]int, 0)
			}
			diag_ca[pos] = append(diag_ca[pos], make([]int, 2))
			diag_ca[pos][len(diag_ca[pos])-1][0] = i
			diag_ca[pos][len(diag_ca[pos])-1][1] = recv

		}
	}

	diag_ca_recv := make([]map[int]int, len(diag_ca))
	for pos := range diag_ca {

		for _, i_and_recv := range diag_ca[pos] {
			recv := i_and_recv[1]
			if diag_ca_recv[pos] == nil {
				diag_ca_recv[pos] = make(map[int]int)
			}
			diag_ca_recv[pos][recv] = 1
		}
	}

	diag_cc := make(map[int][]int)

	for i := 0; i < len(diag[0]); i++ {

		if _, exists := marks[i]; exists {
			continue
		}
		if _, exists := done[i]; exists {
			continue
		}

		if _, exists := traversed[i]; exists {
			continue //fmt.Printf("!!!")
		} else {
			traversed[i] = 1
		}

		if (diag[0][i] <= rem) && (diag[0][i] >= -rem) && (diag[2][i] == _c) {
			if diag_cc[diag[1][i]] == nil {
				diag_cc[diag[1][i]] = make([]int, 0)
			}
			diag_cc[diag[1][i]] = append(diag_cc[diag[1][i]], i)
		}
	}

	diag_cc_idx_sorted := make([]int, 0)
	for k := range diag_cc {
		diag_cc_idx_sorted = append(diag_cc_idx_sorted, k)
	}
	sort.Slice(diag_cc_idx_sorted, func(i, j int) bool {
		return diag_cc_idx_sorted[i] < diag_cc_idx_sorted[j]
	})

	for _, k := range diag_cc_idx_sorted {
		final_merge := false
		for pos := 0; pos < len(diag_ca_recv); pos++ {
			can_merge := true
			for _, i := range diag_cc[k] {
				newDiag := diag[0][i] - 0
				recv := (len(diag[0]) + i - newDiag) % len(diag[0])
				if _, exists := diag_ca_recv[pos][recv]; exists {
					can_merge = false
					break
				}
			}
			if can_merge {
				for _, i := range diag_cc[k] {
					newDiag := diag[0][i] - 0
					recv := (len(diag[0]) + i - newDiag) % len(diag[0])

					if visited[_c][pos][recv] == 1 {
						fmt.Printf("???")
					}

					visited[_c][pos][recv] = 1

					set(musk[_cc], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))

					diag_ca_recv[pos][recv] = 1 // record changes.

					diag[1][i] = pos
					diag[2][i] = _c
				}
				final_merge = true
				break
			}
		}
		if !final_merge {
			pos := len(diag_ca_recv)
			diag_ca_recv = append(diag_ca_recv, make(map[int]int))

			if pos >= len(visited[_c]) {
				visited[_c] = append(visited[_c], make([]int, len(visited[_c][0])))
			}

			for _, i := range diag_cc[k] {
				newDiag := diag[0][i] - 0
				recv := (len(diag[0]) + i - newDiag) % len(diag[0])

				if visited[_c][pos][recv] == 1 {
					fmt.Printf("???")
				}

				visited[_c][pos][recv] = 1

				set(musk[_cc], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
				diag_ca_recv[pos][recv] = 1 // record changes.

				diag[1][i] = pos
				diag[2][i] = _c
			}
		}

	}

	return
}

func SplitConflicts_Test(diag1 [][]int, binary bool) {
	diag := make([][]int, len(diag1))
	for i := 0; i < len(diag1); i++ {
		diag[i] = make([]int, len(diag1[0]))
		copy(diag[i], diag1[i])
	}

	min := minRotAbs(diag[0])
	max := maxRotAbs(diag[0])
	getMargin(diag[0])
	fmt.Println("min", min, " max", max)

	var rot int
	var rem int
	if binary {
		if max > 0 {
			maxlen := bits.Len(uint(max))
			if (1 << maxlen) >= max {
				rot = (1 << (maxlen - 1))
			} else {
				rot = (1 << maxlen)
			}
			rem = rot - 1
		} else {
			rot = 0
			rem = 0
		}

	} else {
		rem = max / 2
		rot = max - rem
	}
	aa_conflicts := make(map[int][]int)
	aa_conflicts_global := make([]map[int][]int, 0)
	aa_conflicts_global_strip := make([]map[int][]int, 0)

	for max := maxRotAbs(diag[0]); max > 0; max = maxRotAbs(diag[0]) {
		if binary {
			if max > 0 {
				maxlen := bits.Len(uint(max))
				if (1 << maxlen) >= max {
					rot = (1 << (maxlen - 1))
				} else {
					rot = (1 << maxlen)
				}
				rem = rot - 1
			} else {
				rot = 0
				rem = 0
			}

		} else {
			rem = max / 2
			rot = max - rem
		}

		visited := make([][][]int, 3)
		for i := 0; i < 3; i++ {
			visited[i] = make([][]int, 1)
			visited[i][0] = make([]int, len(diag[0]))
			for j := 0; j < len(diag[0]); j++ {
				visited[i][0][j] = 0
			}
		}
		aa_conflicts = make(map[int][]int)

		flag := 0

	conflict:
		for i := 0; i < len(diag[0]); i++ {

			if diag[0][i] > rem {

				if flag == 0 && diag[2][i] != _a {
					continue
				}

				newDiag := diag[0][i] - rot
				recv := (len(diag[0]) + i - newDiag) % len(diag[0])

				visited[_a][0][recv] = 1

				if _, exists := aa_conflicts[recv]; exists {
					aa_conflicts[recv] = append(aa_conflicts[recv], diag1[0][i])
				} else {
					aa_conflicts[recv] = make([]int, 1)
					aa_conflicts[recv][0] = diag1[0][i]
				}

				diag[0][i] = newDiag
				diag[2][i] = _a
			} else if flag == 0 && diag[0][i] != 0 {
				diag[2][i] = _c
			}
		}
		if flag == 0 {
			flag = 1
			goto conflict
		}
		aa_conflicts_global = append(aa_conflicts_global, aa_conflicts)
	}
	aa_conflicts_global_strip = make([]map[int][]int, len(aa_conflicts_global))
	for i := range aa_conflicts_global {
		aa_conflicts_global_strip[i] = make(map[int][]int)
		for j := range aa_conflicts_global[i] {
			if len(aa_conflicts_global[i][j]) > 1 {
				aa_conflicts_global_strip[i][j] = make([]int, len(aa_conflicts_global[i][j]))
				copy(aa_conflicts_global_strip[i][j], aa_conflicts_global[i][j])
			}
		}
	}
	return
}

func SplitConflictsV2(diag1 [][]int, marks map[int][]int, done map[int]int, binary bool, group int) (count int, marks_new map[int][]int, done_new map[int]int, Mask [][][]map[string][]int, Rot map[int]int) {

	if (len(done) == len(diag1[0])) && (len(marks) == 0) {
		return 0, nil, nil, nil, nil
	}

	diag := make([][]int, len(diag1))
	for i := 0; i < len(diag1); i++ {
		diag[i] = make([]int, len(diag1[0]))
		if (len(marks) == 0) && (len(done) == 0) {
			copy(diag[i], diag1[i])
		}
	}

	Mask = make([][][]map[string][]int, group+1)

	level := math.MaxInt
	if len(marks) > 0 {
		for i := range marks {
			if marks[i][0] < level {
				level = marks[i][0]
			}
		}
	} else {
		level = 0
	}
	for i := range marks {
		if level == marks[i][0] {
			diag[0][i] = marks[i][1] // load rotations that needs to be done.
			diag[1][i] = marks[i][2] // load the ciphertext index before visit
			diag[2][i] = marks[i][3] // load the ciphertext class before visit
		}
	}

	marks_new = make(map[int][]int, 0)

	done_new = make(map[int]int, 0)

	var a_conflicts []map[int][]int
	if level < math.MaxInt {
		a_conflicts = make([]map[int][]int, level)
	} else {
		panic(errors.New("level >= math.MaxInt"))
	}

	Rot = make(map[int]int, 0)
	/*
		if level < math.MaxInt {
			Rot = make([]int, level)
		}
	*/

	count = 0
	count_cross := 0
	countlv_cross := make(map[int]int)
	min := minRotAbs(diag[0])
	max := maxRotAbs(diag[0])
	getMargin(diag[0])
	fmt.Println("min", min, " max", max)

	var rot int
	var rem int
	if binary {
		if max > 0 {
			maxlen := bits.Len(uint(max))
			if (1 << maxlen) >= max {
				rot = (1 << (maxlen - 1))
			} else {
				rot = (1 << maxlen)
			}
			rem = rot - 1
		} else {
			rot = 0
			rem = 0
		}

	} else {
		rem = max / 2
		rot = max - rem
	}

	for max := maxRotAbs(diag[0]); rot > 0; max = maxRotAbs(diag[0]) {

		if binary {
			if max > 0 {
				maxlen := bits.Len(uint(max))
				if (1 << maxlen) >= max {
					rot = (1 << (maxlen - 1))
				} else {
					rot = (1 << maxlen)
				}
				rem = rot - 1
			} else {
				rot = 0
				rem = 0
			}
		} else {
			rem = max / 2
			rot = max - rem
		}

		a_conflicts = append(a_conflicts, make(map[int][]int))

		Rot[level] = rot

		visited := make([][][]int, 3)
		for i := 0; i < 3; i++ {
			visited[i] = make([][]int, 1)
			visited[i][0] = make([]int, len(diag[0]))
			for j := 0; j < len(diag[0]); j++ {
				visited[i][0][j] = 0
			}
		}

		traversed := make(map[int]int)

		flag := 0

	conflict:
		for i := 0; i < len(diag[0]); i++ {

			if _, exists := done[i]; exists {
				continue
			}
			if _, exists := marks[i]; exists {
				if level < marks[i][0] {
					continue
				}
			}
			if _, exists := marks_new[i]; exists {
				continue
			}

			if diag[0][i] > rem {

				if flag == 0 && diag[2][i] != _a {
					continue
				}

				if _, exists := traversed[i]; exists {
					continue //fmt.Printf("!!!")
				} else {
					traversed[i] = 1
				}

				newDiag := diag[0][i] - rot

				recv := (len(diag[0]) + i - newDiag) % len(diag[0])

				if visited[_a][0][recv] == 1 { // 限制只有1个密文（只看[0]），产生冲突的放到下一轮去

					i2 := a_conflicts[level][recv][0]       // retrieve index
					org_rot2 := a_conflicts[level][recv][1] // retrieve original rotation, this should = diag1[0][i2]

					org_rot := diag1[0][i] // retrieve current element "i"'s original rotation step.

					_, i_exists := marks[i]
					_, i2_exists := marks[i2]

					if i_exists && i2_exists {
						if marks[i][0] == marks[i2][0] && marks[i][0] == level {
							count_cross++
							countlv_cross[level] = 1
						}
					}

					if org_rot <= org_rot2 {
						marks_new[i] = make([]int, 5) // store "i"'s infos for the next network group.
						marks_new[i][0] = level       // store level index
						marks_new[i][1] = diag[0][i]  // store rotations that needs to be done.
						marks_new[i][2] = diag[1][i]  // store the ciphertext index before visit
						marks_new[i][3] = diag[2][i]  // store the ciphertext class before visit

						if _, exists := marks[i]; exists && (marks[i][0] == level) {
							marks_new[i][4] = marks[i][4]
						} else {
							marks_new[i][4] = group
						}

						diag[0][i] = 0

					} else {
						marks_new[i2] = make([]int, 5)
						marks_new[i2][0] = level                       // store level index
						marks_new[i2][1] = diag[0][i2] + rot           // store rotations that needs to be done.
						marks_new[i2][2] = a_conflicts[level][recv][2] // store the ciphertext index before visit
						marks_new[i2][3] = a_conflicts[level][recv][3] // store the ciphertext class before visit

						if _, exists := marks[i2]; exists && (marks[i2][0] == level) {
							marks_new[i2][4] = marks[i2][4]
						} else {
							marks_new[i2][4] = group
						}

						group_idx4i2 := marks_new[i2][4]
						switch marks_new[i2][3] {
						case _a:
							unset(Mask[group_idx4i2][level][_aa], pair(0, marks_new[i2][2]), (len(diag[0])+i2-marks_new[i2][1])%len(diag[0]), len(diag[0]))
						case _c:
							unset(Mask[group_idx4i2][level][_ac], pair(0, marks_new[i2][2]), (len(diag[0])+i2-marks_new[i2][1])%len(diag[0]), len(diag[0]))
						}

						diag[0][i2] = 0

						a_conflicts[level][recv][0] = i           // store input element's index
						a_conflicts[level][recv][1] = diag1[0][i] // store original rotation
						a_conflicts[level][recv][2] = diag[1][i]  // store the ciphertext index before visit
						a_conflicts[level][recv][3] = diag[2][i]  // store the ciphertext class before visit
						if _, exists := marks[i]; exists && (marks[i][0] == level) {
							a_conflicts[level][recv][4] = marks[i][4]
						} else {
							a_conflicts[level][recv][4] = group
						}

						var group_idx = a_conflicts[level][recv][4]
						if len(Mask[group_idx]) < level+1 {
							Mask[group_idx] = append(Mask[group_idx], make([][]map[string][]int, level+1-len(Mask[group_idx]))...)
						}
						if Mask[group_idx][level] == nil {
							Mask[group_idx][level] = make([]map[string][]int, 9)
							for k := range Mask[group_idx][level] {
								Mask[group_idx][level][k] = make(map[string][]int)
							}
						}
						switch diag[2][i] {
						case _a:
							set(Mask[group_idx][level][_aa], pair(0, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
						case _c:
							set(Mask[group_idx][level][_ac], pair(0, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
						}

						diag[0][i] = newDiag
						diag[1][i] = 0 // not necessary...
						diag[2][i] = _a

					}
					count++ // one more conflict solved.
				} else {
					visited[_a][0][recv] = 1
					a_conflicts[level][recv] = make([]int, 5)
					a_conflicts[level][recv][0] = i           // store input element's index
					a_conflicts[level][recv][1] = diag1[0][i] // store original rotation
					a_conflicts[level][recv][2] = diag[1][i]  // store the ciphertext index before visit
					a_conflicts[level][recv][3] = diag[2][i]  // store the ciphertext class before visit
					if _, exists := marks[i]; exists && (marks[i][0] == level) {
						a_conflicts[level][recv][4] = marks[i][4]
					} else {
						a_conflicts[level][recv][4] = group
					}

					var group_idx = a_conflicts[level][recv][4]
					if len(Mask[group_idx]) < level+1 {
						Mask[group_idx] = append(Mask[group_idx], make([][]map[string][]int, level+1-len(Mask[group_idx]))...)
					}
					if Mask[group_idx][level] == nil {
						Mask[group_idx][level] = make([]map[string][]int, 9)
						for k := range Mask[group_idx][level] {
							Mask[group_idx][level][k] = make(map[string][]int)
						}
					}
					switch diag[2][i] {
					case _a:
						set(Mask[group_idx][level][_aa], pair(0, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
					case _c:
						set(Mask[group_idx][level][_ac], pair(0, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
					}

					diag[0][i] = newDiag
					diag[1][i] = 0 // not necessary...
					diag[2][i] = _a
				}

			} else { // if diag[0][i] != 0

				if diag[2][i] != _c && flag == 0 {
					continue
				}
				if diag[2][i] == _c && flag == 1 {
					continue
				}

				if _, exists := traversed[i]; exists {
					continue //fmt.Printf("!!!")
				} else {
					traversed[i] = 1
				}

				newDiag := diag[0][i] - 0
				recv := (len(diag[0]) + i - newDiag) % len(diag[0])
				var pos int
				if diag[2][i] != _c {
					pos = check(&visited[_c], recv)
				} else {

					if rot > 0 {
						if diag[1][i] >= len(visited[_c]) {
							tmp := make([][]int, diag[1][i]+1)
							copy(tmp, visited[_c])
							for i := len(visited[_c]); i < len(tmp); i++ {
								tmp[i] = make([]int, len(visited[_c][0]))
							}
							visited[_c] = tmp
						}
						if visited[_c][diag[1][i]][recv] == 1 {
							fmt.Printf("???")
						}
						visited[_c][diag[1][i]][recv] = 1
						pos = diag[1][i]
					} else {
						pos = check(&visited[_c], recv)
					}
				}

				if len(Mask[group]) < level+1 {
					Mask[group] = append(Mask[group], make([][]map[string][]int, level+1-len(Mask[group]))...)
				}
				if Mask[group][level] == nil {
					Mask[group][level] = make([]map[string][]int, 9)
					for k := range Mask[group][level] {
						Mask[group][level][k] = make(map[string][]int)
					}
				}

				if _, exists := marks[i]; exists && (marks[i][0] == level) {
					panic(errors.New("marks sends an element to the c class")) // no element should be sent to the c class of current group from another previous group.
				}
				switch diag[2][i] {
				case _a:
					set(Mask[group][level][_ca], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
				case _c:
					set(Mask[group][level][_cc], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
				}

				diag[1][i] = pos
				diag[2][i] = _c // do not manage c class's ciphertext in this function.
			}
		}
		if flag == 0 {
			flag = 1
			goto conflict
		}

		count_settled := 0
		count_should := 0
		for i_g := range Mask {
			if (Mask[i_g] != nil) && (len(Mask[i_g]) > level) {
				for t := range Mask[i_g][level] {
					for _, mask := range Mask[i_g][level][t] {
						for _, val := range mask {
							if val == 1 {
								count_settled++
							}
						}
					}
				}
			}
		}

		if (len(marks) == 0) && (len(done) == 0) {
			count_should = len(diag1[0])
		} else {
			for i := range marks {
				if marks[i][0] <= level {
					count_should++
				}
			}
		}
		for i := range marks_new {
			if marks_new[i][0] <= level {
				count_should--
			}
		}
		if count_settled != count_should {
			panic(errors.New("count_settled != count_should"))
		}

		level += 1

		for i := range marks {
			if level == marks[i][0] {

				diag[0][i] = marks[i][1] // load rotations that needs to be done.
				diag[1][i] = marks[i][2] // load the ciphertext index before visit
				diag[2][i] = marks[i][3] // load the ciphertext class before visit
			}
		}

	}

	for i := 0; i < len(diag[0]); i++ {
		done_new[i] = 1
	}
	for i := range marks_new {
		delete(done_new, i)
	}

	count_totalmask := 0
	count_totalmask_perLV := 0
	count_crossmask := 0
	count_crossmaks_perLV := 0
	count_1to1mask := 0
	count_1to1mask_perLV := 0
	count_ac_mask := 0
	count_ac_mask_perLV := 0
	count_ca_mask := 0
	count_ca_mask_perLV := 0
	count_totalnosame := make([]int, level)
	for i_g := range Mask {
		fmt.Printf("Group Cross: %d->%d (Inv: %d->%d)\n", group, i_g, i_g, group)
		for i_lv := range Mask[i_g] {
			count_totalmask_perLV = 0
			count_crossmaks_perLV = 0
			count_1to1mask_perLV = 0
			count_ac_mask_perLV = 0
			count_ca_mask_perLV = 0
			for i_c := range Mask[i_g][i_lv] {
				for k := range Mask[i_g][i_lv][i_c] {
					Mask[i_g][i_lv][i_c][k] = ClearZeroMask(Mask[i_g][i_lv][i_c][k])
					if Mask[i_g][i_lv][i_c][k] != nil {
						count_totalmask++
						count_totalmask_perLV++
						if i_g != group {
							count_crossmask++
							count_crossmaks_perLV++
						}
						if i_c == _ac {
							count_ac_mask++
							count_ac_mask_perLV++
						} else if i_c == _ca {
							count_ca_mask++
							count_ca_mask_perLV++
						}
						s, t := Depair(k)
						if s == t && group == i_g && (i_c == _aa || i_c == _cc) {
							count_1to1mask++
							count_1to1mask_perLV++
						}
					} else {
						delete(Mask[i_g][i_lv][i_c], k)
					}

				}
			}
			fmt.Printf("level %d's total: %d; cross-groups: %d; same_pos: %d, _ac(Inv:_ca): %d, _ca(Inv:_ac):%d\n", i_lv, count_totalmask_perLV, count_crossmaks_perLV, count_1to1mask_perLV, count_ac_mask_perLV, count_ca_mask_perLV)

			count_totalnosame[i_lv] = (count_totalmask_perLV - count_1to1mask_perLV)

		}
	}

	fmt.Printf("total conflict: %d\n", count)
	fmt.Printf("total masks: %d\n", count_totalmask)
	fmt.Printf("total cross-group masks: %d\n", count_crossmask)
	fmt.Printf("same position masks: %d\n", count_1to1mask)
	fmt.Printf("_ac(Inv: _ca) masks: %d\n", count_ac_mask)
	fmt.Printf("_ca(Inv: _ac) masks: %d\n\n", count_ca_mask)
	for i_lv, num := range count_totalnosame {
		fmt.Printf("Level %d's total(no same_pos): %d\n", i_lv, num)
	}
	fmt.Printf("\n")
	return
}

func SplitConflictsV3(diag1 [][]int, marks map[int][]int, done map[int]int, binary bool, group int) (count int, marks_new map[int][]int, done_new map[int]int, Mask [][][]map[string][]int, Rot map[int]int) {
	if (len(done) == len(diag1[0])) && (len(marks) == 0) {
		return 0, nil, nil, nil, nil
	}

	diag := make([][]int, len(diag1))
	for i := 0; i < len(diag1); i++ {
		diag[i] = make([]int, len(diag1[0]))
		if (len(marks) == 0) && (len(done) == 0) {
			copy(diag[i], diag1[i])
		}
	}

	Mask = make([][][]map[string][]int, group+1)

	level := math.MaxInt
	if len(marks) > 0 {
		for i := range marks {
			if marks[i][0] < level {
				level = marks[i][0]
			}
		}
	} else {
		level = 0
	}
	for i := range marks {
		if level == marks[i][0] {
			diag[0][i] = marks[i][1] // load rotations that needs to be done.
			diag[1][i] = marks[i][2] // load the ciphertext index before visit
			diag[2][i] = marks[i][3] // load the ciphertext class before visit
		}
	}

	marks_new = make(map[int][]int, 0)

	done_new = make(map[int]int, 0)

	var a_conflicts []map[int][]int
	if level < math.MaxInt {
		a_conflicts = make([]map[int][]int, level)
	} else {
		panic(errors.New("level >= math.MaxInt"))
	}

	Rot = make(map[int]int, 0)
	/*
		if level < math.MaxInt {
			Rot = make([]int, level)
		}
	*/

	count = 0
	count_cross := 0
	countlv_cross := make(map[int]int)
	min := minRotAbs(diag[0])
	max := maxRotAbs(diag[0])
	getMargin(diag[0])
	fmt.Println("min", min, " max", max)

	var rot int
	var rem int
	if binary {
		if max > 0 {
			maxlen := bits.Len(uint(max))
			if (1 << maxlen) >= max {
				rot = (1 << (maxlen - 1))
			} else {
				rot = (1 << maxlen)
			}
			rem = rot - 1
		} else {
			rot = 0
			rem = 0
		}

	} else {
		rem = max / 2
		rot = max - rem
	}

	visited_num := 1

	for max := maxRotAbs(diag[0]); rot > 0; max = maxRotAbs(diag[0]) {

		if binary {
			if max > 0 {
				maxlen := bits.Len(uint(max))
				if (1 << maxlen) >= max {
					rot = (1 << (maxlen - 1))
				} else {
					rot = (1 << maxlen)
				}
				rem = rot - 1
			} else {
				rot = 0
				rem = 0
			}
		} else {
			rem = max / 2
			rot = max - rem
		}

		a_conflicts = append(a_conflicts, make(map[int][]int))

		Rot[level] = rot

		visited := make([][][]int, 3)
		for i := 0; i < 3; i++ {
			if i == _c {
				visited[i] = make([][]int, visited_num)
			} else {
				visited[i] = make([][]int, 1)
			}
			for j := range visited[i] {
				visited[i][j] = make([]int, len(diag[0]))
			}
		}

		traversed := make(map[int]int)

		flag := 0

	conflict:
		for i := 0; i < len(diag[0]); i++ {

			if _, exists := done[i]; exists {
				continue
			}
			if _, exists := marks[i]; exists {
				if level < marks[i][0] {
					continue
				}
			}
			if _, exists := marks_new[i]; exists {
				continue
			}

			if diag[0][i] > rem {

				if flag == 0 && diag[2][i] != _a {
					continue
				}

				if _, exists := traversed[i]; exists {
					continue //fmt.Printf("!!!")
				} else {
					traversed[i] = 1
				}

				newDiag := diag[0][i] - rot

				recv := (len(diag[0]) + i - newDiag) % len(diag[0])

				if visited[_a][0][recv] == 1 { // The limit is only one ciphertext (only look at [0]), and any conflicts will be carried over to the next round

					i2 := a_conflicts[level][recv][0]       // retrieve index
					org_rot2 := a_conflicts[level][recv][1] // retrieve original rotation, this should = diag1[0][i2]

					org_rot := diag1[0][i] // retrieve current element "i"'s original rotation step.

					_, i_exists := marks[i]
					_, i2_exists := marks[i2]

					if i_exists && i2_exists {
						if marks[i][0] == marks[i2][0] && marks[i][0] == level {
							count_cross++
							countlv_cross[level] = 1
						}
					}

					if org_rot <= org_rot2 {
						marks_new[i] = make([]int, 5) // store "i"'s infos for the next network group.
						marks_new[i][0] = level       // store level index
						marks_new[i][1] = diag[0][i]  // store rotations that needs to be done.
						marks_new[i][2] = diag[1][i]  // store the ciphertext index before visit
						marks_new[i][3] = diag[2][i]  // store the ciphertext class before visit

						if _, exists := marks[i]; exists && (marks[i][0] == level) {
							marks_new[i][4] = marks[i][4]
						} else {
							marks_new[i][4] = group
						}

						diag[0][i] = 0

					} else {
						marks_new[i2] = make([]int, 5)
						marks_new[i2][0] = level                       // store level index
						marks_new[i2][1] = diag[0][i2] + rot           // store rotations that needs to be done.
						marks_new[i2][2] = a_conflicts[level][recv][2] // store the ciphertext index before visit
						marks_new[i2][3] = a_conflicts[level][recv][3] // store the ciphertext class before visit

						if _, exists := marks[i2]; exists && (marks[i2][0] == level) {
							marks_new[i2][4] = marks[i2][4]
						} else {
							marks_new[i2][4] = group
						}

						group_idx4i2 := marks_new[i2][4]
						switch marks_new[i2][3] {
						case _a:
							unset(Mask[group_idx4i2][level][_aa], pair(0, marks_new[i2][2]), (len(diag[0])+i2-marks_new[i2][1])%len(diag[0]), len(diag[0]))
						case _c:
							unset(Mask[group_idx4i2][level][_ac], pair(0, marks_new[i2][2]), (len(diag[0])+i2-marks_new[i2][1])%len(diag[0]), len(diag[0]))
						}

						diag[0][i2] = 0

						a_conflicts[level][recv][0] = i           // store input element's index
						a_conflicts[level][recv][1] = diag1[0][i] // store original rotation
						a_conflicts[level][recv][2] = diag[1][i]  // store the ciphertext index before visit
						a_conflicts[level][recv][3] = diag[2][i]  // store the ciphertext class before visit
						if _, exists := marks[i]; exists && (marks[i][0] == level) {
							a_conflicts[level][recv][4] = marks[i][4]
						} else {
							a_conflicts[level][recv][4] = group
						}

						var group_idx = a_conflicts[level][recv][4]
						if len(Mask[group_idx]) < level+1 {
							Mask[group_idx] = append(Mask[group_idx], make([][]map[string][]int, level+1-len(Mask[group_idx]))...)
						}
						if Mask[group_idx][level] == nil {
							Mask[group_idx][level] = make([]map[string][]int, 9)
							for k := range Mask[group_idx][level] {
								Mask[group_idx][level][k] = make(map[string][]int)
							}
						}
						switch diag[2][i] {
						case _a:
							set(Mask[group_idx][level][_aa], pair(0, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
						case _c:
							set(Mask[group_idx][level][_ac], pair(0, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
						}

						diag[0][i] = newDiag
						diag[1][i] = 0 // not necessary...
						diag[2][i] = _a

					}
					count++ // one more conflict solved.
				} else {
					visited[_a][0][recv] = 1
					a_conflicts[level][recv] = make([]int, 5)
					a_conflicts[level][recv][0] = i           // store input element's index
					a_conflicts[level][recv][1] = diag1[0][i] // store original rotation
					a_conflicts[level][recv][2] = diag[1][i]  // store the ciphertext index before visit
					a_conflicts[level][recv][3] = diag[2][i]  // store the ciphertext class before visit
					if _, exists := marks[i]; exists && (marks[i][0] == level) {
						a_conflicts[level][recv][4] = marks[i][4]
					} else {
						a_conflicts[level][recv][4] = group
					}

					var group_idx = a_conflicts[level][recv][4]
					if len(Mask[group_idx]) < level+1 {
						Mask[group_idx] = append(Mask[group_idx], make([][]map[string][]int, level+1-len(Mask[group_idx]))...)
					}
					if Mask[group_idx][level] == nil {
						Mask[group_idx][level] = make([]map[string][]int, 9)
						for k := range Mask[group_idx][level] {
							Mask[group_idx][level][k] = make(map[string][]int)
						}
					}
					switch diag[2][i] {
					case _a:
						set(Mask[group_idx][level][_aa], pair(0, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
					case _c:
						set(Mask[group_idx][level][_ac], pair(0, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
					}

					diag[0][i] = newDiag
					diag[1][i] = 0 // not necessary...
					diag[2][i] = _a
				}

			} else { // if diag[0][i] != 0

				if diag[2][i] != _c && flag == 0 {
					continue
				}
				if diag[2][i] == _c && flag == 1 {
					continue
				}

				if _, exists := traversed[i]; exists {
					continue //fmt.Printf("!!!")
				} else {
					traversed[i] = 1
				}

				newDiag := diag[0][i] - 0
				recv := (len(diag[0]) + i - newDiag) % len(diag[0])
				var pos int
				if diag[2][i] != _c {
					if rot > 0 {
						visited[_c][visited_num-1][recv] = 1
						pos = visited_num - 1
					} else {
						pos = check(&visited[_c], recv) // might need to have stronger assumption
					}
				} else {

					if rot > 0 {
						if diag[1][i] >= len(visited[_c]) {
							tmp := make([][]int, diag[1][i]+1)
							copy(tmp, visited[_c])
							for i := len(visited[_c]); i < len(tmp); i++ {
								tmp[i] = make([]int, len(visited[_c][0]))
							}
							visited[_c] = tmp
						}
						if visited[_c][diag[1][i]][recv] == 1 {
							fmt.Printf("???")
						}
						visited[_c][diag[1][i]][recv] = 1
						pos = diag[1][i]
					} else {
						pos = check(&visited[_c], recv) // might need to have stronger assumption
					}
				}

				if len(Mask[group]) < level+1 {
					Mask[group] = append(Mask[group], make([][]map[string][]int, level+1-len(Mask[group]))...)
				}
				if Mask[group][level] == nil {
					Mask[group][level] = make([]map[string][]int, 9)
					for k := range Mask[group][level] {
						Mask[group][level][k] = make(map[string][]int)
					}
				}

				if _, exists := marks[i]; exists && (marks[i][0] == level) {
					panic(errors.New("marks sends an element to the c class")) // no element should be sent to the c class of current group from another previous group.
				}
				switch diag[2][i] {
				case _a:
					set(Mask[group][level][_ca], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
				case _c:
					set(Mask[group][level][_cc], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
				}

				diag[1][i] = pos
				diag[2][i] = _c // do not manage c class's ciphertext in this function.
			}
		}
		if flag == 0 {
			flag = 1
			goto conflict
		}

		count_settled := 0
		count_should := 0
		for i_g := range Mask {
			if (Mask[i_g] != nil) && (len(Mask[i_g]) > level) {
				for t := range Mask[i_g][level] {
					for _, mask := range Mask[i_g][level][t] {
						for _, val := range mask {
							if val == 1 {
								count_settled++
							}
						}
					}
				}
			}
		}

		if (len(marks) == 0) && (len(done) == 0) {
			count_should = len(diag1[0])
		} else {
			for i := range marks {
				if marks[i][0] <= level {
					count_should++
				}
			}
		}
		for i := range marks_new {
			if marks_new[i][0] <= level {
				count_should--
			}
		}
		if count_settled != count_should {
			panic(errors.New("count_settled != count_should"))
		}

		level += 1

		visited_num += 1

		for i := range marks {
			if level == marks[i][0] {

				diag[0][i] = marks[i][1] // load rotations that needs to be done.
				diag[1][i] = marks[i][2] // load the ciphertext index before visit
				diag[2][i] = marks[i][3] // load the ciphertext class before visit
			}
		}

	}

	for i := 0; i < len(diag[0]); i++ {
		done_new[i] = 1
	}
	for i := range marks_new {
		delete(done_new, i)
	}

	count_totalmask := 0
	count_totalmask_perLV := 0
	count_crossmask := 0
	count_crossmaks_perLV := 0
	count_1to1mask := 0
	count_1to1mask_perLV := 0
	count_ac_mask := 0
	count_ac_mask_perLV := 0
	count_ca_mask := 0
	count_ca_mask_perLV := 0
	count_totalnosame := make([]int, level)
	for i_g := range Mask {
		fmt.Printf("Group Cross: %d->%d (Inv: %d->%d)\n", group, i_g, i_g, group)
		for i_lv := range Mask[i_g] {
			count_totalmask_perLV = 0
			count_crossmaks_perLV = 0
			count_1to1mask_perLV = 0
			count_ac_mask_perLV = 0
			count_ca_mask_perLV = 0
			for i_c := range Mask[i_g][i_lv] {
				for k := range Mask[i_g][i_lv][i_c] {
					Mask[i_g][i_lv][i_c][k] = ClearZeroMask(Mask[i_g][i_lv][i_c][k])
					if Mask[i_g][i_lv][i_c][k] != nil {
						count_totalmask++
						count_totalmask_perLV++
						if i_g != group {
							count_crossmask++
							count_crossmaks_perLV++
						}
						if i_c == _ac {
							count_ac_mask++
							count_ac_mask_perLV++
						} else if i_c == _ca {
							count_ca_mask++
							count_ca_mask_perLV++
						}
						s, t := Depair(k)
						if s == t && group == i_g && (i_c == _aa || i_c == _cc) {
							count_1to1mask++
							count_1to1mask_perLV++
						}
					} else {
						delete(Mask[i_g][i_lv][i_c], k)
					}

				}
			}
			fmt.Printf("level %d's total: %d; cross-groups: %d; same_pos: %d, _ac(Inv:_ca): %d, _ca(Inv:_ac):%d\n", i_lv, count_totalmask_perLV, count_crossmaks_perLV, count_1to1mask_perLV, count_ac_mask_perLV, count_ca_mask_perLV)

			count_totalnosame[i_lv] = (count_totalmask_perLV - count_1to1mask_perLV)

		}
	}

	fmt.Printf("total conflict: %d\n", count)
	fmt.Printf("total masks: %d\n", count_totalmask)
	fmt.Printf("total cross-group masks: %d\n", count_crossmask)
	fmt.Printf("same position masks: %d\n", count_1to1mask)
	fmt.Printf("_ac(Inv: _ca) masks: %d\n", count_ac_mask)
	fmt.Printf("_ca(Inv: _ac) masks: %d\n\n", count_ca_mask)
	for i_lv, num := range count_totalnosame {
		fmt.Printf("Level %d's total(no same_pos): %d\n", i_lv, num)
	}
	fmt.Printf("\n")

	return // count, marks_new, done_new
}

func DecomposeV5(diag [][]int, marks map[int]int, done map[int]int, binary bool) (musk []map[string][]int, rot int, visited [][][]int) {
	max := maxRotAbs(diag[0])

	var rem int
	if binary {
		if max > 0 {
			maxlen := bits.Len(uint(max))
			if (1 << maxlen) >= max {
				rot = (1 << (maxlen - 1))
			} else {
				rot = (1 << maxlen)
			}
			rem = rot - 1
		} else {
			rot = 0
			rem = 0
		}
	} else {
		rem = max / 2
		rot = max - rem
	}

	visited = make([][][]int, 3)
	for i := 0; i < 3; i++ {
		visited[i] = make([][]int, 1)
		visited[i][0] = make([]int, len(diag[0]))
		for j := 0; j < len(diag[0]); j++ {
			visited[i][0][j] = 0
		}
	}

	musk = make([]map[string][]int, 9)
	for i := 0; i < 9; i++ {
		musk[i] = make(map[string][]int)
	}

	traversed := make(map[int]int)

	for i := 0; i < len(diag[0]); i++ {

		if _, exists := marks[i]; exists {
			continue
		}

		if _, exists := done[i]; exists {
			continue
		}

		if diag[0][i] > rem {

			traversed[i] = 1

			newDiag := diag[0][i] - rot
			recv := (len(diag[0]) + i - newDiag) % len(diag[0])
			pos := check(&visited[_a], recv)

			switch diag[2][i] {
			case _a:
				set(musk[_aa], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			case _c:
				set(musk[_ac], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
			}
			diag[0][i] = newDiag
			diag[1][i] = pos
			diag[2][i] = _a
		}
	}

	diag_ca_recv := make([]map[int]int, 0)

	diag_cc := make(map[int][]int)

	for i := 0; i < len(diag[0]); i++ {

		if _, exists := marks[i]; exists {
			continue
		}
		if _, exists := done[i]; exists {
			continue
		}

		if (diag[0][i] <= rem) && (diag[0][i] >= -rem) && (diag[2][i] == _c) {

			if _, exists := traversed[i]; exists {
				continue //fmt.Printf("!!!")
			} else {
				traversed[i] = 1
			}

			if diag_cc[diag[1][i]] == nil {
				diag_cc[diag[1][i]] = make([]int, 0)
			}
			diag_cc[diag[1][i]] = append(diag_cc[diag[1][i]], i)
		}
	}

	diag_cc_idx_sorted := make([]int, 0)
	for k := range diag_cc {
		diag_cc_idx_sorted = append(diag_cc_idx_sorted, k)
	}
	sort.Slice(diag_cc_idx_sorted, func(i, j int) bool {
		return diag_cc_idx_sorted[i] < diag_cc_idx_sorted[j]
	})

	for _, k := range diag_cc_idx_sorted {
		final_merge := false
		for pos := 0; pos < len(diag_ca_recv); pos++ {
			can_merge := true
			for _, i := range diag_cc[k] {
				newDiag := diag[0][i] - 0
				recv := (len(diag[0]) + i - newDiag) % len(diag[0])
				if _, exists := diag_ca_recv[pos][recv]; exists {
					can_merge = false
					break
				}
			}
			if can_merge {
				for _, i := range diag_cc[k] {
					newDiag := diag[0][i] - 0
					recv := (len(diag[0]) + i - newDiag) % len(diag[0])

					if visited[_c][pos][recv] == 1 {
						fmt.Printf("???")
					}

					visited[_c][pos][recv] = 1

					set(musk[_cc], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))

					diag_ca_recv[pos][recv] = 1 // record changes.

					diag[1][i] = pos
					diag[2][i] = _c
				}
				final_merge = true
				break
			}
		}
		if !final_merge {
			pos := len(diag_ca_recv)
			diag_ca_recv = append(diag_ca_recv, make(map[int]int))

			if pos >= len(visited[_c]) {
				visited[_c] = append(visited[_c], make([]int, len(visited[_c][0])))
			}

			for _, i := range diag_cc[k] {
				newDiag := diag[0][i] - 0
				recv := (len(diag[0]) + i - newDiag) % len(diag[0])

				if visited[_c][pos][recv] == 1 {
					fmt.Printf("???")
				}

				visited[_c][pos][recv] = 1

				set(musk[_cc], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))
				diag_ca_recv[pos][recv] = 1 // record changes.

				diag[1][i] = pos
				diag[2][i] = _c
			}
		}

	}

	for i := 0; i < len(diag[0]); i++ {
		if _, exists := marks[i]; exists {
			continue
		}
		if _, exists := done[i]; exists {
			continue
		}

		if _, exists := traversed[i]; exists {
			continue //fmt.Printf("!!!")
		} else {
			traversed[i] = 1
		}

		traversed[i] = 1

		newDiag := diag[0][i] - 0
		recv := (len(diag[0]) + i - newDiag) % len(diag[0])
		pos := check(&visited[_c], recv)

		set(musk[_ca], pair(pos, diag[1][i]), (len(diag[0])+i-diag[0][i])%len(diag[0]), len(diag[0]))

		diag[1][i] = pos
		diag[2][i] = _c

	}

	return
}

func PrepareMasksPtInv(RRot []map[int]int, MMask [][][][]map[string][]int, params ckks.Parameters, Maxlevel int) (lv_begin, lv_end, inc int, RRotInv []map[int]int, MMaskInv [][][][]map[string][]int, MMaskpt [][][][]map[string]*rlwe.Plaintext) {
	if Maxlevel > params.MaxLevel() || Maxlevel < 0 {
		panic("cannot prepare MasksPtInv: invalid Maxlevel")
	}

	lv_max := 0
	for m := range RRot {
		if lv_max < len(RRot[m])-1 {
			lv_max = len(RRot[m]) - 1
		}
	}

	lv_begin = 1
	lv_end = lv_max
	inc = 1

	if (Maxlevel - lv_end + lv_begin) < 0 {
		panic("cannot prepare MasksPtInv: Maxlevel smaller than level budget")
	}

	var MMask_tmp [][][][]map[string][]int
	var RRot_tmp []map[int]int

	MMask_tmp = make([][][][]map[string][]int, len(MMask))
	RRot_tmp = make([]map[int]int, len(RRot))
	for m := range RRot {
		RRot_tmp[m] = make(map[int]int)
		for k := range RRot[m] {
			if k < lv_max {
				RRot_tmp[m][k+1] = -RRot[m][k]
			}
		}
		RRot_tmp[m][0] = 0
	}
	RRotInv = RRot_tmp

	for m := range MMask {

		for i_g := range MMask[m] {
			if len(MMask_tmp[i_g])-1 < m {
				MMask_tmp[i_g] = append(MMask_tmp[i_g], make([][][]map[string][]int, m-len(MMask_tmp[i_g])+1)...)
			}
			for lv := range MMask[m][i_g] {
				for cls := range MMask[m][i_g][lv] {

					if len(MMask_tmp[i_g][m])-1 < lv {
						MMask_tmp[i_g][m] = append(MMask_tmp[i_g][m], make([][]map[string][]int, lv-len(MMask_tmp[i_g][m])+1)...)
					}

					var cls_inv int
					if cls == _ac {
						cls_inv = _ca
					} else if cls == _ca {
						cls_inv = _ac
					} else {
						cls_inv = cls
					}

					if len(MMask_tmp[i_g][m][lv])-1 < cls_inv {
						MMask_tmp[i_g][m][lv] = append(MMask_tmp[i_g][m][lv], make([]map[string][]int, cls_inv-len(MMask_tmp[i_g][m][lv])+1)...)
					}
					if MMask_tmp[i_g][m][lv][cls_inv] == nil {
						MMask_tmp[i_g][m][lv][cls_inv] = make(map[string][]int)
					}

					for k, v := range MMask[m][i_g][lv][cls] {
						k_inv := InversePair(k)
						MMask_tmp[i_g][m][lv][cls_inv][k_inv] = make([]int, len(v))
						copy(MMask_tmp[i_g][m][lv][cls_inv][k_inv], v)
					}

				}
			}
		}
	}
	MMaskInv = MMask_tmp

	encoder := ckks.NewEncoder(params)
	MMaskpt = make([][][][]map[string]*rlwe.Plaintext, len(MMask_tmp))
	for m := range MMask_tmp {
		MMaskpt[m] = make([][][]map[string]*rlwe.Plaintext, len(MMask_tmp[m]))
		for i_g := range MMask_tmp[m] {
			MMaskpt[m][i_g] = make([][]map[string]*rlwe.Plaintext, len(MMask_tmp[m][i_g]))
			for lv := range MMask_tmp[m][i_g] {
				if lv == 0 {
					continue
				}
				var ecd_level int
				ecd_level = Maxlevel - lv + lv_begin
				ecd_level = ecd_level - (lv_end - (len(MMask_tmp[i_g][i_g]) - 1))
				var scale = params.NewScale(params.RingQ().Modulus[ecd_level])
				if len(MMask_tmp[m][i_g][lv]) == 0 {
					MMaskpt[m][i_g][lv] = nil
				} else {
					MMaskpt[m][i_g][lv] = make([]map[string]*rlwe.Plaintext, len(MMask_tmp[m][i_g][lv]))
				}
				for cls := range MMask_tmp[m][i_g][lv] {
					MMaskpt[m][i_g][lv][cls] = make(map[string]*rlwe.Plaintext)
					for k, v := range MMask_tmp[m][i_g][lv][cls] {
						vFloat := make([]float64, len(v))
						for i := 0; i < len(v); i++ {
							vFloat[i] = float64(v[i])
						}
						v_pt := encoder.EncodeMontNew(vFloat, ecd_level, scale, params.LogSlots())
						MMaskpt[m][i_g][lv][cls][k] = v_pt
					}
				}
			}
		}
	}

	return

}

func PrepareMasksPtInv_ForCollapse(RRot []map[int]int, MMask [][][][]map[string][]int, params ckks.Parameters, Maxlevel int, BotCollpaseDepth int) (lv_begin, lv_end, inc int, RRotInv []map[int]int, MMaskInv [][][][]map[string][]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, ecd_levelEachGroup []map[int]int) {

	if Maxlevel > params.MaxLevel() || Maxlevel < 0 {
		panic("cannot prepare MasksPtInv: invalid Maxlevel")
	}

	lv_max := 0
	for m := range RRot {
		if lv_max < len(RRot[m])-1 {
			lv_max = len(RRot[m]) - 1
		}
	}

	lv_begin = 1
	lv_end = lv_max
	inc = 1

	/*
		if (Maxlevel - lv_end + lv_begin) < 0 {
			panic("cannot prepare MasksPtInv: Maxlevel smaller than level budget")
		}
	*/

	var MMask_tmp [][][][]map[string][]int
	var RRot_tmp []map[int]int

	MMask_tmp = make([][][][]map[string][]int, len(MMask))
	RRot_tmp = make([]map[int]int, len(RRot))
	for m := range RRot {
		RRot_tmp[m] = make(map[int]int)
		for k := range RRot[m] {
			if k < lv_max {
				RRot_tmp[m][k+1] = -RRot[m][k]
			}
		}
		RRot_tmp[m][0] = 0
	}
	RRotInv = RRot_tmp

	for m := range MMask {

		for i_g := range MMask[m] {
			if len(MMask_tmp[i_g])-1 < m {
				MMask_tmp[i_g] = append(MMask_tmp[i_g], make([][][]map[string][]int, m-len(MMask_tmp[i_g])+1)...)
			}
			for lv := range MMask[m][i_g] {
				for cls := range MMask[m][i_g][lv] {

					if len(MMask_tmp[i_g][m])-1 < lv {
						MMask_tmp[i_g][m] = append(MMask_tmp[i_g][m], make([][]map[string][]int, lv-len(MMask_tmp[i_g][m])+1)...)
					}

					var cls_inv int
					if cls == _ac {
						cls_inv = _ca
					} else if cls == _ca {
						cls_inv = _ac
					} else {
						cls_inv = cls
					}

					if len(MMask_tmp[i_g][m][lv])-1 < cls_inv {
						MMask_tmp[i_g][m][lv] = append(MMask_tmp[i_g][m][lv], make([]map[string][]int, cls_inv-len(MMask_tmp[i_g][m][lv])+1)...)
					}
					if MMask_tmp[i_g][m][lv][cls_inv] == nil {
						MMask_tmp[i_g][m][lv][cls_inv] = make(map[string][]int)
					}

					for k, v := range MMask[m][i_g][lv][cls] {
						k_inv := InversePair(k)
						MMask_tmp[i_g][m][lv][cls_inv][k_inv] = make([]int, len(v))
						copy(MMask_tmp[i_g][m][lv][cls_inv][k_inv], v)
					}

				}
			}
		}
	}
	MMaskInv = MMask_tmp

	var lv_beforeCollapse int
	for lv := lv_end; lv >= 0; lv-- {
		if math.Abs(float64(RRotInv[len(RRotInv)-1][lv])) >= float64((int(1) << (BotCollpaseDepth + 1))) {
			lv_beforeCollapse = lv
			break
		}
	}

	new_lvidx := make(map[int]int)
	lv_offset := (-lv_beforeCollapse) + (lv_end - (BotCollpaseDepth + 1))
	for lv := lv_begin; lv <= lv_end; lv++ { // for lv := range RRotInv[len(RRotInv)-1] {
		if lv <= lv_beforeCollapse {
			new_lvidx[lv] = (lv - lv_beforeCollapse) + (lv_end - (BotCollpaseDepth + 1)) //  - 1)
		}
	}

	for m := range MMask_tmp {
		if len(MMask_tmp[m][len(RRotInv)-1]) == 0 {
			continue
		}
		tmpBuff := make([][]map[string][]int, len(MMask_tmp[m][len(RRotInv)-1])+lv_offset)
		for lv := range MMask_tmp[m][len(RRotInv)-1] {
			if len(MMask_tmp[m][len(RRotInv)-1][lv]) == 0 {
				continue
			}
			if _, exists := new_lvidx[lv]; exists {
				src_standby_id := 0
				mapping := MMask_tmp[m][m][lv][_ac]
				var s, t int
				for k, _ := range mapping {
					fmt.Sscanf(k, "%d:%d", &s, &t)
				}
				src_standby_id = t

				tmpBuff[new_lvidx[lv]] = MMask_tmp[m][len(RRotInv)-1][lv]
				if lv_offset > 0 && (m < (len(RRotInv) - 1)) {
					if _, ok := MMask_tmp[m][len(RRotInv)-1][lv][_aa]["0:0"]; ok {
						mapping_str_new := fmt.Sprintf("%d:%d", src_standby_id, 0)
						tmpBuff[new_lvidx[lv]][_ca][mapping_str_new] = MMask_tmp[m][len(RRotInv)-1][lv][_aa]["0:0"]
						tmpBuff[new_lvidx[lv]][_aa] = make(map[string][]int) // set empty.
					}
				}
			}
		}
		MMask_tmp[m][len(RRotInv)-1] = tmpBuff
	}
	tmpRRot := make(map[int]int)
	for lv := range RRotInv[len(RRotInv)-1] {
		tmpRRot[new_lvidx[lv]] = RRotInv[len(RRotInv)-1][lv]
	}
	RRotInv[len(RRotInv)-1] = tmpRRot

	encoder := ckks.NewEncoder(params)
	MMaskpt = make([][][][]map[string]*rlwe.Plaintext, len(MMask_tmp))
	ecd_levelEachGroup = make([]map[int]int, len(RRotInv)) // ecd_levelEachGroup := make([]map[int]int, len(RRotInv))
	for m := range MMask_tmp {
		MMaskpt[m] = make([][][]map[string]*rlwe.Plaintext, len(MMask_tmp[m]))
		for i_g := range MMask_tmp[m] {
			MMaskpt[m][i_g] = make([][]map[string]*rlwe.Plaintext, len(MMask_tmp[m][i_g]))
			if ecd_levelEachGroup[i_g] == nil {
				ecd_levelEachGroup[i_g] = make(map[int]int)
			}
			for lv := range MMask_tmp[m][i_g] {
				if lv == 0 {
					continue
				}
				var ecd_level int

				/*
					count_smaller := 0
					count_bigger := 0
					for rot_lv := range RRotInv[i_g] {
						if math.Abs(float64(RRotInv[i_g][rot_lv])) < float64((int(1) << (BotCollpaseDepth + 1))) {
							count_smaller++
						}
						if rot_lv >= lv {
							if math.Abs(float64(RRotInv[i_g][rot_lv])) >= float64((int(1) << (BotCollpaseDepth + 1))) {
								count_bigger++
							}
						}
					}
					_ = count_smaller
					_, exists_rot := RRotInv[i_g][lv]
					if !exists_rot {
						ecd_level = Maxlevel - lv + lv_begin - ((lv_end - BotCollpaseDepth - 1) - (count_bigger + (lv - lv_begin))) // Maxlevel - lv + lv_begin - ((lv_end - BotCollpaseDepth) - (count_bigger))
						ecd_level = ecd_level + 1
					} else {

						if -RRotInv[i_g][lv] >= (1 << (BotCollpaseDepth + 1)) {
							ecd_level = Maxlevel - lv + lv_begin - ((lv_end - BotCollpaseDepth - 1) - (count_bigger + (lv - lv_begin))) // Maxlevel - lv + lv_begin - ((lv_end - BotCollpaseDepth) - (count_bigger))
							if i_g >= 5 {
								ecd_level = Maxlevel - lv + lv_begin - ((lv_end - BotCollpaseDepth - 1) - (count_bigger + (lv - lv_begin))) // Maxlevel - lv + lv_begin - ((lv_end - BotCollpaseDepth) - (count_bigger))
							}
						} else {
							ecd_level = Maxlevel - lv + lv_begin
							ecd_level = ecd_level - (lv_end - (len(MMask_tmp[i_g][i_g]) - 1))
						}

					}
				*/

				ecd_level = Maxlevel - lv + lv_begin
				if _, exists := ecd_levelEachGroup[i_g][lv]; exists {
					if ecd_levelEachGroup[i_g][lv] != ecd_level {
						panic("should be consistent")
					}
				} else {
					ecd_levelEachGroup[i_g][lv] = ecd_level
				}

				var scale rlwe.Scale
				if len(MMask_tmp[m][i_g][lv]) == 0 || ecd_level < 0 {
					MMaskpt[m][i_g][lv] = nil
					continue
				} else {
					MMaskpt[m][i_g][lv] = make([]map[string]*rlwe.Plaintext, len(MMask_tmp[m][i_g][lv]))
					scale = params.NewScale(params.RingQ().Modulus[ecd_level])
				}
				for cls := range MMask_tmp[m][i_g][lv] {
					MMaskpt[m][i_g][lv][cls] = make(map[string]*rlwe.Plaintext)
					for k, v := range MMask_tmp[m][i_g][lv][cls] {
						vFloat := make([]float64, len(v))
						for i := 0; i < len(v); i++ {
							vFloat[i] = float64(v[i])
						}
						v_pt := encoder.EncodeMontNew(vFloat, ecd_level, scale, params.LogSlots())
						MMaskpt[m][i_g][lv][cls][k] = v_pt
					}
				}
			}
		}
	}

	return

}

func MultiGroupNetworkTopDownNewV3_pt(ctIn []int, RRot []map[int]int, MMaskpt [][][][]map[string][]int, lv_begin, lv_end, inc, lv_collapse int, org_p []int, lv_collapse_bottom int, pre, nex [][]map[int][]int, nex_cnt [][]map[int]int) (ctOut []int, nexcollapse [][]map[int]map[int][]int, nexcollapseB [][]map[int]map[int][]int) {
	var _a = 0
	var _c = 1

	var _aa = 0
	var _ac = 1
	var _ca = 2
	var _cc = 3

	copy(pre[0][_a][0], ctIn)
	pre[0][_c][0] = ctIn

	var ctTmp1 []int
	var ctTmp2 []int
	var _x, _y, s, t int
	var lv_end_rlt int
	var visitOrder = [4]int{_ac, _aa, _ca, _cc}
	var cnt int

	for lv := lv_begin; lv <= lv_end; lv += inc {

		for m := range RRot {

			for k := range pre[m][_a] {
				if step, exists := RRot[m][lv]; exists {
					ctTmp1 = pre[m][_a][k]

					ctTmp1_tmp := make([]int, len(ctTmp1))
					for i := 0; i < len(ctTmp1); i++ {
						ctTmp1_tmp[(i-step)%len(ctTmp1)] = ctTmp1[i]
					}
					copy(ctTmp1, ctTmp1_tmp)

				}
			}
		}

		enable_level_collapsing_bottom := true
		BotcollapseMulNum := 0
		EachEntryCount := make(map[int][]string)
		if enable_level_collapsing_bottom {
			if lv_end-lv == lv_collapse_bottom+1 {
				nexcollapseB = make([][]map[int]map[int][]int, len(pre))
				totalDiffRotsB := make(map[int]int)
				for i_g := range pre {
					nexcollapseB[i_g] = make([]map[int]map[int][]int, len(pre[i_g]))
					for x := range pre[i_g] {
						nexcollapseB[i_g][x] = make(map[int]map[int][]int)
						for t := range pre[i_g][x] {
							nexcollapseB[i_g][x][t] = make(map[int][]int, len(pre[i_g][x][t]))
							for idx, e := range pre[i_g][x][t] {
								dist := 0
								if e > 0 && e <= len(pre[i_g][x][t]) {
									dest_idx := 0
									for destId, destE := range org_p {
										if destE == e-1 {
											dest_idx = destId
										}
									}
									dist = idx - dest_idx // idx - org_p[e-1]

									if dist > 0 {
										dist = dist - len(pre[i_g][x][t])
									}

									if math.Abs(float64(dist)) >= math.Pow(2, float64(lv_collapse_bottom+1)) {
										continue
									}
									/*
										if _, exists := EachEntryCount[e-1]; exists {
											continue
										} else {
											EachEntryCount[e-1] = 1
										}
									*/

									totalDiffRotsB[dist] = 1
									pos_beforerot := idx
									if _, exists := nexcollapseB[i_g][x][t][dist]; exists {
										nexcollapseB[i_g][x][t][dist][pos_beforerot] = 1
									} else {
										nexcollapseB[i_g][x][t][dist] = make([]int, len(pre[i_g][x][t]))
										nexcollapseB[i_g][x][t][dist][pos_beforerot] = 1
										BotcollapseMulNum++
									}
									if _, exists := EachEntryCount[e-1]; exists {
										EachEntryCount[e-1] = append(EachEntryCount[e-1], fmt.Sprintf("pre[%d][%d][%d] idx[%d],dist[%d]", i_g, x, t, idx, dist))
									} else {
										EachEntryCount[e-1] = make([]string, 1)
										EachEntryCount[e-1][0] = fmt.Sprintf("pre[%d][%d][%d] idx[%d],dist[%d]", i_g, x, t, idx, dist)
									}
								}
							}
						}
					}
				}
				_ = nexcollapseB
				ctBuff := make(map[int][]int, len(totalDiffRotsB))
				cntBuff := make(map[int][]int, len(totalDiffRotsB))
				for i := range totalDiffRotsB {
					ctBuff[i] = make([]int, len(ctIn))
					cntBuff[i] = make([]int, len(ctIn))
				}
				for i_g := range nexcollapseB {
					for t := range nexcollapseB[i_g] {
						for idx := range nexcollapseB[i_g][t] {
							if len(nexcollapseB[i_g][t][idx]) == 0 {
								continue
							}
							for m_id, mask := range nexcollapseB[i_g][t][idx] {

								for i := range mask {
									ctBuff[m_id][i] += (mask[i] * pre[i_g][t][idx][i])
									if mask[i] >= 1 {
										cntBuff[m_id][i]++
									}
									if cntBuff[m_id][i] > 1 {
										panic("Error here, should not be > 1")
									}
								}

							}
						}
					}
				}
				ctOutBot := make([]int, len(ctIn))
				for m_id := range ctBuff {
					for i := range ctBuff[m_id] {
						rotated_i := (i - m_id + len(ctBuff[m_id])) % len(ctBuff[m_id])
						ctOutBot[rotated_i] += ctBuff[m_id][i]
					}
				}
				ctOut = ctOutBot
				return

			}
		}

		for i_g := len(RRot) - 1; i_g >= 0; i_g-- {

			lv_end_rlt = len(MMaskpt[i_g][i_g]) - 1

			if lv < lv_end_rlt {
			}

			for m := i_g; m >= 0; m-- {
				if len(MMaskpt[m][i_g]) <= lv || MMaskpt[m][i_g][lv] == nil {
					continue
				}
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
								for i := 0; i < len(ctTmp2); i++ {
									ctTmp2[i] = (ctTmp1[i] * v[i])
								}
							} else {
								for i := 0; i < len(ctTmp2); i++ {
									ctTmp2[i] += (ctTmp1[i] * v[i])
								}
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac && lv == lv_end_rlt-1) || (cls == _cc && lv == lv_end_rlt-1):
							/*
								if nex[i_g][_y][t] == nil {
									nex[i_g][_y][t] = NewCiphertext(eval.params, ctTmp1.Degree(), ecd_level)
								}
							*/
							ctTmp2 = nex[i_g][_y][t]
							for i := 0; i < len(ctTmp2); i++ {
								ctTmp2[i] = (ctTmp1[i] * v[i])
							}
							nex_cnt[i_g][_y][t]++

						case (cls == _ac):
							if lv < lv_end_rlt-1 {
								copy(nex[i_g][_y][t], ctTmp1)
							} else if lv == lv_end_rlt {
								nex[i_g][_x][t] = ctTmp1 // we only do this once, so no worry for modification.
							}

						case (cls == _cc):
							if lv < lv_end_rlt-1 {
								nex[i_g][_y][t] = ctTmp1 // we only do this once per ciphertext, so no worry for modification.
							} else if lv == lv_end_rlt {
								if cnt == 0 {
									ctOut = make([]int, len(ctTmp1))
									copy(ctOut, ctTmp1)
								} else {
									ctTmp2 = ctOut
									/*
										if ctTmp1.Level() != ctTmp2.Level() {
											panic("???")
										}
									*/
									for i := 0; i < len(ctTmp2); i++ {
										ctTmp2[i] = ctTmp1[i] + ctTmp2[i]
									}
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
						_ = cnt_tmp
						delete(nex_cnt[m][k], t)
					}
				}
				pre[m] = nex[m]
			}
		}
		enable_level_collapsing := true
		if enable_level_collapsing {
			if (lv-lv_begin == lv_collapse) && (lv_collapse > 0) {
				nexcollapse = make([][]map[int]map[int][]int, len(nex)) // last map[int]int: denotes different rots dist in a node.
				totalDiffRots := make(map[int]int)
				for i_g := range nex {
					nexcollapse[i_g] = make([]map[int]map[int][]int, len(nex[i_g]))
					for _x := range nex[i_g] {
						nexcollapse[i_g][_x] = make(map[int]map[int][]int, len(nex[i_g][_x]))
						for t := range nex[i_g][_x] {
							nexcollapse[i_g][_x][t] = make(map[int][]int, len(nex[i_g][_x][t]))
							for idx, e := range nex[i_g][_x][t] {
								idxfrom1 := idx + 1
								dist := 0
								if e > 0 && e <= len(nex[i_g][_x][t]) {
									dist = e - idxfrom1 // (e - idxfrom1) % (len(nex[i_g][_x][t]) + 1) // dist = (e - idx) % len(nex[i_g][_x][t])
									if dist > 0 {       // manually mod n into (-n,0]
										dist = dist - len(nex[i_g][_x][t])
									}

									totalDiffRots[dist] = 1

									if _, exists := nexcollapse[i_g][_x][t][dist]; exists {
										org_idx := e - 1                                        // (dist + e) % len(nex[i_g][_x][t])
										pos_afterrot := (org_idx - dist) % len(nex[i_g][_x][t]) // this should be equal to idx: e-1-(e-idx-1)=idx
										nexcollapse[i_g][_x][t][dist][pos_afterrot] = 1
									} else {
										nexcollapse[i_g][_x][t][dist] = make([]int, len(nex[i_g][_x][t]))
										org_idx := e - 1                                        // (dist + e) % len(nex[i_g][_x][t])
										pos_afterrot := (org_idx - dist) % len(nex[i_g][_x][t]) // this should be equal to idx: e-1-(e-idx-1)=idx
										nexcollapse[i_g][_x][t][dist][pos_afterrot] = 1
									}
								}
							}
						}
					}
				}
				ct_DiffRots := make(map[int][]int)
				for dist := range totalDiffRots {
					ct_DiffRots[dist] = make([]int, len(ctIn))
					copy(ct_DiffRots[dist], ctIn)
					ct_tmp := make([]int, len(ctIn))
					for i := range ct_DiffRots[dist] {
						ct_tmp[(i-dist)%len(ctIn)] = ct_DiffRots[dist][i]
					}
					ct_DiffRots[dist] = ct_tmp
				}
				for i_g := range nex {
					for _x := range nex[i_g] {
						for t := range nex[i_g][_x] {
							for idx := range nex[i_g][_x][t] {
								nex[i_g][_x][t][idx] = 0
							}
						}
					}
				}
				for i_g := range pre {
					for _x := range pre[i_g] {
						for t := range pre[i_g][_x] {
							for idx := range pre[i_g][_x][t] {
								pre[i_g][_x][t][idx] = 0
							}
						}
					}
				}
				for i_g := range nexcollapse {
					for _x := range nexcollapse[i_g] {
						for t := range nexcollapse[i_g][_x] {
							cnt4make := 0
							for dist := range nexcollapse[i_g][_x][t] {
								if cnt4make == 0 {
									nex[i_g][_x][t] = make([]int, len(ctIn))
									cnt4make++
								}
								empty := true
								for idx, mask_value := range nexcollapse[i_g][_x][t][dist] {
									if mask_value != 0 {
										empty = false
									}
									nex[i_g][_x][t][idx] += (mask_value * ct_DiffRots[dist][idx])
								}
								if empty {
									fmt.Printf("Found one empty mask.\n")
								}
							}

							if len(nexcollapse[i_g][_x][t]) >= 1 {
								fmt.Printf("Aggregated nex[%d][%d][%d] \n", i_g, _x, t)
								/*
									fmt.Printf("[")
									for e := 0; e < len(nex[i_g][_x][t]); e++ {
										fmt.Printf("%d ", nex[i_g][_x][t][e])
									}
									fmt.Printf("]\n")
									fmt.Printf("\n")
								*/
							}

						}
					}
					pre[i_g] = nex[i_g]
				}
			}

		}

		/*
			enable_level_collapsing_bottom := true
			BotcollapseMulNum := 0
			EachEntryCount := make(map[int]int)
			if enable_level_collapsing_bottom {
				if lv_end-lv == lv_collapse_bottom {
					nexcollapseB = make([][]map[int]map[int][]int, len(nex))
					totalDiffRotsB := make(map[int]int)
					for i_g := range nex {
						nexcollapseB[i_g] = make([]map[int]map[int][]int, len(nex[i_g]))
						for x := range nex[i_g] {
							nexcollapseB[i_g][x] = make(map[int]map[int][]int)
							for t := range nex[i_g][x] {
								nexcollapseB[i_g][x][t] = make(map[int][]int, len(nex[i_g][x][t]))
								for idx, e := range nex[i_g][x][t] {
									dist := 0
									if e > 0 && e <= len(nex[i_g][x][t]) {
										dest_idx := 0
										for destId, destE := range org_p {
											if destE == e-1 {
												dest_idx = destId
											}
										}
										dist = idx - dest_idx // idx - org_p[e-1]

										if dist > 0 {
											dist = dist - len(nex[i_g][x][t])
										}

										if math.Abs(float64(dist)) >= math.Pow(2, float64(lv_collapse_bottom+1)) {
											continue
										}
										if _, exists := EachEntryCount[e-1]; exists {
											continue
										} else {
											EachEntryCount[e-1] = 1
										}

										totalDiffRotsB[dist] = 1
										pos_beforerot := idx
										if _, exists := nexcollapseB[i_g][x][t][dist]; exists {
											nexcollapseB[i_g][x][t][dist][pos_beforerot] = 1
										} else {
											nexcollapseB[i_g][x][t][dist] = make([]int, len(nex[i_g][x][t]))
											nexcollapseB[i_g][x][t][dist][pos_beforerot] = 1
											BotcollapseMulNum++
										}
									}
								}
							}
						}
					}
					_ = nexcollapseB
					ctBuff := make(map[int][]int, len(totalDiffRotsB))
					cntBuff := make(map[int][]int, len(totalDiffRotsB))
					for i := range totalDiffRotsB {
						ctBuff[i] = make([]int, len(ctIn))
						cntBuff[i] = make([]int, len(ctIn))
					}
					for i_g := range nexcollapseB {
						for t := range nexcollapseB[i_g] {
							for idx := range nexcollapseB[i_g][t] {
								if len(nexcollapseB[i_g][t][idx]) == 0 {
									continue
								}
								for m_id, mask := range nexcollapseB[i_g][t][idx] {

									for i := range mask {
										ctBuff[m_id][i] += (mask[i] * nex[i_g][t][idx][i])
										if mask[i] >= 1 {
											cntBuff[m_id][i]++
										}
										if cntBuff[m_id][i] > 1 {
											panic("Error here, should not be > 1")
										}
									}

								}
							}
						}
					}
					ctOutBot := make([]int, len(ctIn))
					for m_id := range ctBuff {
						for i := range ctBuff[m_id] {
							rotated_i := (i - m_id + len(ctBuff[m_id])) % len(ctBuff[m_id])
							ctOutBot[rotated_i] += ctBuff[m_id][i]
						}
					}
					ctOut = ctOutBot
					return

				}
			}
		*/

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
						fmt.Printf("group %d, level %d nex[%d][%d]\n", i_g, lv, s, t)
						for i := range nex[i_g][s][t] {
							if nex[i_g][s][t][i] == 7077 {
								fmt.Printf("has 7077 at pos %d\n", i)
							}
						}
					}
				}
			}
		*/

	}

	var last_lv int
	cnt = 0
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
		/*
			if ctTmp1.Level() != ecd_level {
				panic("????")
			}
		*/
		for i := 0; i < len(ctOut); i++ {
			ctOut[i] += ctTmp1[i]
		}
		cnt++
	}
	return
}

func PrepareNexAndPre_pt(MMaskpt [][][][]map[string][]int, n int) (pre, nex [][]map[int][]int, nex_cnt [][]map[int]int) {

	var _a = 0
	var _c = 1
	maxNum := len(MMaskpt[0][0])

	pre = make([][]map[int][]int, len(MMaskpt))
	nex = make([][]map[int][]int, len(MMaskpt))

	nex_cnt = make([][]map[int]int, len(MMaskpt))
	for i := range pre {
		pre[i] = make([]map[int][]int, 2)
		nex[i] = make([]map[int][]int, 2)
		nex_cnt[i] = make([]map[int]int, 2)

		pre[i][_a] = make(map[int][]int)
		pre[i][_c] = make(map[int][]int)
		nex[i][_a] = make(map[int][]int)
		nex[i][_c] = make(map[int][]int)
		nex_cnt[i][_a] = make(map[int]int)
		nex_cnt[i][_c] = make(map[int]int)

		pre[i][_a][0] = make([]int, n)
		nex[i][_a][0] = make([]int, n)

		for k := 0; k < maxNum; k++ {
			pre[i][_c][k] = make([]int, n)
			nex[i][_c][k] = make([]int, n)
		}
	}

	return
}

/*
func ArrangeAndClassifyMasks(RRot []map[int]int, MMaskpt [][][][]map[string]*rlwe.Plaintext, aaMask [][][]map[]) {

}
*/

func Partition(matrix [][]float64, subrow int, subcol int) (blocks [][][][]float64) {
	row := len(matrix)
	col := len(matrix[0])
	if (row%subrow != 0) || (col%subcol != 0) {
		panic(errors.New("(row mod subrow != 0) || (col mod subcol != 0)"))
	}
	blocks = make([][][][]float64, row/subrow)
	for i := range blocks {
		blocks[i] = make([][][]float64, col/subcol)
		for j := range blocks[i] {
			blocks[i][j] = make([][]float64, subrow)
			for k := range blocks[i][j] {
				blocks[i][j][k] = make([]float64, subcol)
			}
		}
	}
	for i := range matrix {
		for j := range matrix[i] {
			blocks[i/subrow][j/subcol][i%subrow][j%subcol] = matrix[i][j]
		}
	}

	return
}

func PartitionAndEncodeInterspersedVec(matrix [][]float64, subrow int, subcol int) (vec []float64) {
	mat := PartitionAndEncodeInterspersedMat(matrix, subrow, subcol)
	vec = make([]float64, 0)
	for i := range mat {
		for j := range mat[i] {
			vec = append(vec, mat[i][j]...)
		}
	}
	return
}

func PartitionAndEncodeInterspersedMat(matrix [][]float64, subrow int, subcol int) (mat [][][]float64) {
	blocks := Partition(matrix, subrow, subcol)
	mat = make([][][]float64, subrow)
	for i := range mat {
		mat[i] = make([][]float64, subcol)
		for j := range mat[i] {
			mat[i][j] = make([]float64, 0)
			for rb := range blocks {
				for cb := range blocks[rb] {
					mat[i][j] = append(mat[i][j], blocks[rb][cb][i][j])
				}
			}
		}
	}
	return
}

func PartitionAndEncodeInterspersedVec_ColOrder(matrix [][]float64, subrow int, subcol int) (vec []float64) {
	mat := PartitionAndEncodeInterspersedMat_ColOrder(matrix, subrow, subcol)
	vec = make([]float64, 0)
	for i := range mat {
		for j := range mat[i] {
			vec = append(vec, mat[i][j]...)
		}
	}
	return
}

func PartitionAndEncodeInterspersedMat_ColOrder(matrix [][]float64, subrow int, subcol int) (mat [][][]float64) {
	blocks := Partition(matrix, subrow, subcol)
	mat = make([][][]float64, subrow)
	for i := range mat {
		mat[i] = make([][]float64, subcol)
		for j := range mat[i] {
			mat[i][j] = make([]float64, 0)
			for cb := 0; cb < len(blocks[0]); cb++ {
				for rb := range blocks {
					mat[i][j] = append(mat[i][j], blocks[rb][cb][i][j])
				}
			}
		}
	}
	return
}

func DecodeInterspersedVec(vec []float64, subrow int, subcol int, numblock_perrow int, numblock_percol int) [][]float64 {
	mat := make([][][]float64, subrow)
	if (len(vec) % (subcol * subrow)) != 0 {
		panic(errors.New("len(vec) mod (subcol*subrow) != 0"))
	}
	if (numblock_percol*subrow)*(numblock_perrow*subcol) != len(vec) {
		panic(errors.New("(numblock_percol*subrow)*(numblock_perrow*subcol) != len(vec)"))
	}

	num := len(vec) / (subcol * subrow)

	for i := range mat {
		mat[i] = make([][]float64, subcol)
		for j := range mat[i] {
			mat[i][j] = make([]float64, num)
		}
	}
	for i := range vec {
		actual_index := i / num
		actual_rank := i % num
		mat[actual_index/subcol][actual_index%subcol][actual_rank] = vec[i]
	}

	rslt := make([][]float64, numblock_percol*subrow)
	for i := range rslt {
		rslt[i] = make([]float64, numblock_perrow*subcol)
	}
	for s_row := range mat {
		for s_col := range mat[s_row] {
			for rank := range mat[s_row][s_col] {
				idx_row := ((rank)/numblock_perrow)*subrow + s_row
				idx_col := ((rank)%numblock_perrow)*subcol + s_col

				rslt[idx_row][idx_col] = mat[s_row][s_col][rank]
			}
		}
	}
	return rslt
}

func DecodeInterspersedVec_ColOrder(vec []float64, subrow int, subcol int, numblock_perrow int, numblock_percol int) [][]float64 {
	mat := make([][][]float64, subrow)
	if (len(vec) % (subcol * subrow)) != 0 {
		panic(errors.New("len(vec) mod (subcol*subrow) != 0"))
	}
	if (numblock_percol*subrow)*(numblock_perrow*subcol) != len(vec) {
		panic(errors.New("(numblock_percol*subrow)*(numblock_perrow*subcol) != len(vec)"))
	}

	num := len(vec) / (subcol * subrow)

	for i := range mat {
		mat[i] = make([][]float64, subcol)
		for j := range mat[i] {
			mat[i][j] = make([]float64, num)
		}
	}
	for i := range vec {
		actual_index := i / num
		actual_rank := i % num
		mat[actual_index/subcol][actual_index%subcol][actual_rank] = vec[i]
	}

	rslt := make([][]float64, numblock_percol*subrow)
	for i := range rslt {
		rslt[i] = make([]float64, numblock_perrow*subcol)
	}
	for s_row := range mat {
		for s_col := range mat[s_row] {
			for rank := range mat[s_row][s_col] {

				idx_row := ((rank)%numblock_percol)*subrow + s_row
				idx_col := ((rank)/numblock_percol)*subcol + s_col

				rslt[idx_row][idx_col] = mat[s_row][s_col][rank]
			}
		}
	}
	return rslt
}

func MatrixTranspose(matrix [][]float64) [][]float64 {
	n := len(matrix[0])
	m := len(matrix)
	rslt := make([][]float64, n)
	for i := range rslt {
		rslt[i] = make([]float64, m)
	}
	for i := range matrix {
		for j := range matrix[i] {
			rslt[j][i] = matrix[i][j]
		}
	}
	return rslt
}

func MatrixReplication(matrix [][]float64, times int, axis int) [][]float64 {

	rslt := make([][]float64, 0)
	if (axis != 0) && (axis != 1) {
		panic(errors.New("axis != 0 || axis != 1"))
	}
	/*
		if times < 0 { // if times < 1 {
			panic(errors.New("times < 1"))
		}
	*/

	switch axis {
	case 1:
		rslt = make([][]float64, len(matrix))
		for i := range matrix {
			rslt[i] = make([]float64, len(matrix[i]))
			copy(rslt[i], matrix[i])
		}
		for i := 0; i < times; i++ {
			rslt = append(rslt, matrix...)
		}

	case 0:
		rslt = make([][]float64, len(matrix))
		for i := range matrix {
			rslt[i] = make([]float64, len(matrix[i]))
			copy(rslt[i], matrix[i])
			for j := 0; j < times; j++ {
				rslt[i] = append(rslt[i], matrix[i]...)
			}
		}
	}
	return rslt
}
