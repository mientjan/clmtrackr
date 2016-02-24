export interface Scoring
{
	size: number[];
	bias: number;
	coef: number[];
}

export interface Path
{
	normal: any[];
	vertices: number[][];
}

export interface PatchModel
{
	patchType: string;
	weights: number[][][];
	numPatches: number;
	patchSize: number[];
	canvasSize: number[];
}

export interface ShapeModel
{
	eigenVectors: number[][];
	numEvalues: number;
	eigenValues: number[];
	numPtsPerSample: number;
	nonRegularizedVectors: number[];
	meanShape: number[][];
}

export interface Hints
{
	rightEye: number[];
	leftEye: number[];
	nose: number[];
}

export interface IFaceTrackingModel
{
	scoring: Scoring;
	path: Path;
	patchModel: PatchModel;
	shapeModel: ShapeModel;
	hints: Hints;
}

