package ckksconfig

import (
	"encoding/json"
	"os"
)

type Config struct {
	Mass            float64   `json:"m"`
	Length          float64   `json:"l"`
	Gravity         float64   `json:"g"`
	DT              float64   `json:"dt"`
	N               int       `json:"N"`
	QDiag           []float64 `json:"QDiag"`
	RDiag           []float64 `json:"RDiag"`
	QfScale         float64   `json:"QfScale"`
	X0              []float64 `json:"x0"`
	ThetaMax        float64   `json:"thetaMax"`
	OmegaMax        float64   `json:"omegaMax"`
	UMax            float64   `json:"uMax"`
	Sigma0          float64   `json:"sigma0"`
	LambdaParam     float64   `json:"lambda"`
	LogN            int       `json:"logN"`
	LogQ            []int     `json:"logQ"`
	LogP            []int     `json:"logP"`
	LogDefaultScale int       `json:"logDefaultScale"`
	K               int       `json:"K"`
	NWorkers        int       `json:"nWorkers"`
	T               int       `json:"T"`
	TSteps          int       `json:"TSteps"`
	ChebOrder       int       `json:"chebOrder"`
	ChebBound       float64   `json:"chebBound"`
	ChebEta         float64   `json:"chebEta"`
	ChebClip        bool      `json:"chebClip"`
}

func Load(path string) (Config, bool) {
	var cfg Config
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, false
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		panic(err)
	}
	return cfg, true
}
