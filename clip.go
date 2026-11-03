package clip

import "github.com/oxzjh/cv"

type IText interface {
	Extract(string) ([]float32, error)
	Close()
}

type IImage interface {
	Extract(*cv.Mat) ([]float32, error)
	ExtractFile(string) ([]float32, error)
	Close()
}

type Clip struct {
	txt IText
	img IImage
}

func (c *Clip) ExtractText(text string) ([]float32, error) {
	return c.txt.Extract(text)
}

func (c *Clip) ExtractImage(m *cv.Mat) ([]float32, error) {
	return c.img.Extract(m)
}

func (c *Clip) ExtractImageFile(file string) ([]float32, error) {
	return c.img.ExtractFile(file)
}

func (c *Clip) Close() {
	c.txt.Close()
	c.img.Close()
}

func New(txt IText, img IImage) *Clip {
	return &Clip{txt, img}
}
