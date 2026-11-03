package ort

import (
	"errors"

	"github.com/oxzjh/cv"
	"github.com/oxzjh/ort"
	"github.com/oxzjh/vector"
)

type Image struct {
	engine     *ort.Ort
	width      int
	height     int
	outputDims int
}

func (img *Image) Extract(m *cv.Mat) ([]float32, error) {
	resized, _ := m.ResizeWithBorder(img.width, img.height, cv.INTER_CUBIC, cv.BORDER_CONSTANT, cv.NewScalar(114, 114, 114, 0))
	inputBlob := resized.BlobFromImage(1/255., cv.Size{}, cv.NewScalar(122.7709383, 116.7460125, 104.09373615, 0), true, false)
	resized.Free()
	inputBlob.Divide(cv.NewScalar(0.26862954, 0.26130258, 0.27577711, 1))
	defer inputBlob.Free()
	result, err := img.engine.Infer(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), ort.TypeFloat)
	if err != nil {
		return nil, err
	}
	defer result.Free()
	feature, err := result.GetFloat32(0, img.outputDims, true)
	if err != nil {
		return nil, err
	}
	vector.NormalizeSelf(feature)
	return feature, nil
}

func (img *Image) ExtractFile(file string) ([]float32, error) {
	m := cv.IMRead(file, cv.FLAG_COLOR)
	defer m.Free()
	if m.Cols() < 4 || m.Rows() < 4 {
		return nil, errors.New("read image failed")
	}
	return img.Extract(m)
}

func (img *Image) Close() {
	if img.engine != nil {
		img.engine.Close()
		img.engine = nil
	}
}

func NewImage(model string, threads int) (*Image, error) {
	engine, err := ort.New(model, "image_clip", threads)
	if err != nil {
		return nil, err
	}
	engine.InputDims[0][0] = 1
	return &Image{engine, int(engine.InputDims[0][3]), int(engine.InputDims[0][2]), int(engine.OutputDims[0][1])}, nil
}
