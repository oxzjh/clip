package clip

import (
	"errors"
	"io"
	"os"
	"unsafe"

	"github.com/oxzjh/tokenizer"
	"github.com/oxzjh/vector"
)

func ReadProjection(proj string) (map[int]map[int]float32, error) {
	f, err := os.Open(proj)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 128)
	if _, err = f.Read(header); err != nil {
		return nil, err
	}

	for i, v := range header {
		if v == '\n' {
			f.Seek(int64(i)+1, io.SeekStart)
		}
	}

	b, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	if len(b) != 4*512*512 {
		return nil, errors.New("read projection failed")
	}

	p := unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)>>2)
	pm := make(map[int]map[int]float32, 512)
	for i := 0; i < 512; i++ {
		pm[i] = make(map[int]float32, 512)
		for j := 0; j < 512; j++ {
			pm[i][j] = p[i*512+j]
		}
	}

	return pm, nil
}

func GetMaxIndex(tokens tokenizer.Tokens) int {
	var (
		maxToken int
		maxIndex int
	)
	for i := range tokens {
		if v := tokens[i].ID; v > maxToken {
			maxToken = v
			maxIndex = i
		}
	}
	return maxIndex
}

func GetFeature(projection map[int]map[int]float32, extracted []float32) []float32 {
	feature := make([]float32, len(extracted))
	for i := 0; i < 512; i++ {
		for j := 0; j < 512; j++ {
			feature[i] += extracted[j] * projection[j][i]
		}
	}
	vector.NormalizeSelf(feature)
	return feature
}
