package auxiliary_io

/*
 * This part of package auxiliary_io aims to implement IO of csv file. It has been tested that library "encoding/csv"
 * has worse performance than simply using library "bufio", since it does not support loacting and reading a specific
 * row (neither can a Scanner of bufio, but the latter is faster). Although seeking some specific offset of bytes in the file
 * is available with *os.File object, correctly locating the starting byte of some line in a csv file is not so easy.
 * So here we use a Scanner of bufio to scan the file until we reach the target line.
 */

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
)

func GetOneCSVSubblock(csvfilepath string, rowIdx int, colIdx int, rowSize int, colSize int) (subblock [][]string, err error) {
	var csvfile *os.File
	csvfile, err = os.Open(csvfilepath)
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}
	defer csvfile.Close()
	scanner := bufio.NewScanner(csvfile)
	scanner.Split(bufio.ScanLines)
	for i := 0; i < rowIdx; i++ {
		scanner.Scan()
	}
	subblock = make([][]string, rowSize)
	for i := 0; i < rowSize; i++ {
		subblock[i] = make([]string, colSize)
		scanner.Scan()
		line_str := scanner.Text()
		fields := strings.Split(line_str, ",")
		copy(subblock[i], fields[colIdx:colIdx+colSize])
	}
	return
}

func ExportToCSV(data [][]float64, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)

	for _, row := range data {
		strRow := make([]string, len(row))
		for i, val := range row {
			strRow[i] = fmt.Sprintf("%f", val)
		}
		if err := writer.Write(strRow); err != nil {
			return err
		}
	}

	writer.Flush()

	return nil
}

func ReadMatFromCSV(filePath string) ([][]float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error reading CSV: %v", err)
	}

	size := len(records)
	if size == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}

	matrix := make([][]float64, size)
	for i := range matrix {
		matrix[i] = make([]float64, size)
	}

	for i, record := range records {
		if len(record) < size {
			continue
		}

		for j, value := range record {
			if value == "" {
				continue
			}

			num, err := strconv.Atoi(value) // strconv.Atoi(value)
			if err != nil {
				return nil, fmt.Errorf("error converting value at row %d, column %d: %v", i, j, err)
			}

			matrix[i][j] = float64(num)
		}
	}

	return matrix, nil
}

func WriteMatToCSV(matrix [][]float64, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("error creating file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, row := range matrix {
		record := make([]string, len(row))
		for j, value := range row {
			if value == 0 {
				record[j] = "" // Write empty string for zeros
			} else {
				record[j] = strconv.Itoa(int(value))
			}
		}

		if err := writer.Write(record); err != nil {
			return fmt.Errorf("error writing record to CSV: %v", err)
		}
	}

	return nil
}

func ReadCSVTo2DFloat(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("error reading header row: %v", err)
	}

	var data [][]float64
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		row := make([]float64, len(record))
		for i, v := range record {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing value %q at row %d, column %d: %v", v, len(data)+1, i+1, err)
			}
			row[i] = f
		}

		data = append(data, row)
	}

	return data, nil
}
