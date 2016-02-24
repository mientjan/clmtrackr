import {IFaceTrackingModel} from "./interface/IFaceTrackingModel";
class ClmTracker
{
	model:IFaceTrackingModel;
	patchType:string;
	numPatches:number;
	patchSize:number;

	constructor(model:IFaceTrackingModel)
	{
		this.model = model;

		this.patchType = model.patchModel.patchType;
		this.numPatches = model.patchModel.numPatches;
		this.patchSize = model.patchModel.patchSize[0];
	}

	public init():void
	{

			if (patchType == "MOSSE") {
				searchWindow = patchSize;
			} else {
				searchWindow = params.searchWindow;
			}

			numParameters = model.shapeModel.numEvalues;
			modelWidth = model.patchModel.canvasSize[0];
			modelHeight = model.patchModel.canvasSize[1];

			// set up canvas to work on
			sketchCanvas = document.createElement('canvas');
			sketchCC = sketchCanvas.getContext('2d');

			sketchW = sketchCanvas.width = modelWidth + (searchWindow-1) + patchSize-1;
			sketchH = sketchCanvas.height = modelHeight + (searchWindow-1) + patchSize-1;

			if (model.hints && mosseFilter && left_eye_filter && right_eye_filter && nose_filter) {
				//var mossef_lefteye = new mosseFilter({drawResponse : document.getElementById('overlay2')});
				mossef_lefteye = new mosseFilter();
				mossef_lefteye.load(left_eye_filter);
				//var mossef_righteye = new mosseFilter({drawResponse : document.getElementById('overlay2')});
				mossef_righteye = new mosseFilter();
				mossef_righteye.load(right_eye_filter);
				//var mossef_nose = new mosseFilter({drawResponse : document.getElementById('overlay2')});
				mossef_nose = new mosseFilter();
				mossef_nose.load(nose_filter);
			} else {
				console.log("MOSSE filters not found, using rough approximation for initialization.");
			}

			// load eigenvectors
			eigenVectors = numeric.rep([numPatches*2,numParameters],0.0);
			for (var i = 0;i < numPatches*2;i++) {
				for (var j = 0;j < numParameters;j++) {
					eigenVectors[i][j] = model.shapeModel.eigenVectors[i][j];
				}
			}

			// load mean shape
			for (var i = 0; i < numPatches;i++) {
				meanShape[i] = [model.shapeModel.meanShape[i][0], model.shapeModel.meanShape[i][1]];
			}

			// get max and mins, width and height of meanshape
			msxmax = msymax = 0;
			msxmin = msymin = 1000000;
			for (var i = 0;i < numPatches;i++) {
				if (meanShape[i][0] < msxmin) msxmin = meanShape[i][0];
				if (meanShape[i][1] < msymin) msymin = meanShape[i][1];
				if (meanShape[i][0] > msxmax) msxmax = meanShape[i][0];
				if (meanShape[i][1] > msymax) msymax = meanShape[i][1];
			}
			msmodelwidth = msxmax-msxmin;
			msmodelheight = msymax-msymin;

			// get scoringweights if they exist
			if (model.scoring) {
				scoringWeights = new Float64Array(model.scoring.coef);
				scoringBias = model.scoring.bias;
				scoringCanvas.width = model.scoring.size[0];
				scoringCanvas.height = model.scoring.size[1];
			}

			// load eigenvalues
			eigenValues = model.shapeModel.eigenValues;

			weights = model.patchModel.weights;
			biases = model.patchModel.bias;

			// precalculate gaussianPriorDiagonal
			gaussianPD = numeric.rep([numParameters+4, numParameters+4],0);
			// set values and append manual inverse
			for (var i = 0;i < numParameters;i++) {
				if (model.shapeModel.nonRegularizedVectors.indexOf(i) >= 0) {
					gaussianPD[i+4][i+4] = 1/10000000;
				} else {
					gaussianPD[i+4][i+4] = 1/eigenValues[i];
				}
			}

			for (var i = 0;i < numParameters+4;i++) {
				currentParameters[i] = 0;
			}

			if (patchType == "SVM") {
				var webGLContext;
				var webGLTestCanvas = document.createElement('canvas');
				if (window.WebGLRenderingContext) {
					webGLContext = webGLTestCanvas.getContext('webgl') || webGLTestCanvas.getContext('experimental-webgl');
					if (!webGLContext || !webGLContext.getExtension('OES_texture_float')) {
						webGLContext = null;
					}
				}

				if (webGLContext && params.useWebGL && (typeof(webglFilter) !== "undefined")) {
					webglFi = new webglFilter();
					try {
						webglFi.init(weights, biases, numPatches, searchWindow+patchSize-1, searchWindow+patchSize-1, patchSize, patchSize);
						if ('lbp' in weights) lbpInit = true;
						if ('sobel' in weights) sobelInit = true;
					}
					catch(err) {
						alert("There was a problem setting up webGL programs, falling back to slightly slower javascript version. :(");
						webglFi = undefined;
						svmFi = new svmFilter();
						svmFi.init(weights['raw'], biases['raw'], numPatches, patchSize, searchWindow);
					}
				} else if (typeof(svmFilter) !== "undefined") {
					// use fft convolution if no webGL is available
					svmFi = new svmFilter();
					svmFi.init(weights['raw'], biases['raw'], numPatches, patchSize, searchWindow);
				} else {
					throw "Could not initiate filters, please make sure that svmfilter.js or svmfilter_conv_js.js is loaded."
				}
			} else if (patchType == "MOSSE") {
				mosseCalc = new mosseFilterResponses();
				mosseCalc.init(weights, numPatches, patchSize, patchSize);
			}

			if (patchType == "SVM") {
				pw = pl = patchSize+searchWindow-1;
			} else {
				pw = pl = searchWindow;
			}
			pdataLength = pw*pl;
			halfSearchWindow = (searchWindow-1)/2;
			responsePixels = searchWindow*searchWindow;
			if(typeof Float64Array !== 'undefined') {
				vecProbs = new Float64Array(responsePixels);
				for (var i = 0;i < numPatches;i++) {
					patches[i] = new Float64Array(pdataLength);
				}
			} else {
				vecProbs = new Array(responsePixels);
				for (var i = 0;i < numPatches;i++) {
					patches[i] = new Array(pdataLength);
				}
			}

			for (var i = 0;i < numPatches;i++) {
				learningRate[i] = 1.0;
				prevCostFunc[i] = 0.0;
			}

			if (params.weightPoints) {
				// weighting of points
				pointWeights = [];
				for (var i = 0;i < numPatches;i++) {
					if (i in params.weightPoints) {
						pointWeights[(i*2)] = params.weightPoints[i];
						pointWeights[(i*2)+1] = params.weightPoints[i];
					} else {
						pointWeights[(i*2)] = 1;
						pointWeights[(i*2)+1] = 1;
					}
				}
				pointWeights = numeric.diag(pointWeights);
			}
	}

	public track(element, box) {

	var scaling, translateX, translateY, rotation;
	var croppedPatches = [];
	var ptch, px, py;

	if (first) {
		// do viola-jones on canvas to get initial guess, if we don't have any points
		var gi = this.getInitialPosition(element, box);
		if (!gi) {
			// send an event on no face found
			var evt = document.createEvent("Event");
			evt.initEvent("clmtrackrNotFound", true, true);
			document.dispatchEvent(evt)

			return false;
		}
		scaling = gi[0];
		rotation = gi[1];
		translateX = gi[2];
		translateY = gi[3];

		first = false;
	} else {
		facecheck_count += 1;

		if (params.constantVelocity) {
			// calculate where to get patches via constant velocity prediction
			if (previousParameters.length >= 2) {
				for (var i = 0;i < currentParameters.length;i++) {
					currentParameters[i] = (relaxation)*previousParameters[1][i] + (1-relaxation)*((2*previousParameters[1][i]) - previousParameters[0][i]);
					//currentParameters[i] = (3*previousParameters[2][i]) - (3*previousParameters[1][i]) + previousParameters[0][i];
				}
			}
		}

		// change translation, rotation and scale parameters
		rotation = halfPI - Math.atan((currentParameters[0]+1)/currentParameters[1]);
		if (rotation > halfPI) {
			rotation -= Math.PI;
		}
		scaling = currentParameters[1] / Math.sin(rotation);
		translateX = currentParameters[2];
		translateY = currentParameters[3];
	}

	// copy canvas to a new dirty canvas
	sketchCC.save();

	// clear canvas
	sketchCC.clearRect(0, 0, sketchW, sketchH);

	sketchCC.scale(1/scaling, 1/scaling);
	sketchCC.rotate(-rotation);
	sketchCC.translate(-translateX, -translateY);

	sketchCC.drawImage(element, 0, 0, element.width, element.height);

	sketchCC.restore();
	//	get cropped images around new points based on model parameters (not scaled and translated)
	var patchPositions = calculatePositions(currentParameters, false);

	// check whether tracking is ok
	if (scoringWeights && (facecheck_count % 10 == 0)) {
		if (!checkTracking()) {
			// reset all parameters
			first = true;
			scoringHistory = [];
			for (var i = 0;i < currentParameters.length;i++) {
				currentParameters[i] = 0;
				previousParameters = [];
			}

			// send event to signal that tracking was lost
			var evt = document.createEvent("Event");
			evt.initEvent("clmtrackrLost", true, true);
			document.dispatchEvent(evt)

			return false;
		}
	}


	var pdata, pmatrix, grayscaleColor;
	for (var i = 0; i < numPatches; i++) {
		px = patchPositions[i][0]-(pw/2);
		py = patchPositions[i][1]-(pl/2);
		ptch = sketchCC.getImageData(Math.round(px), Math.round(py), pw, pl);
		pdata = ptch.data;

		// convert to grayscale
		pmatrix = patches[i];
		for (var j = 0;j < pdataLength;j++) {
			grayscaleColor = pdata[j*4]*0.3 + pdata[(j*4)+1]*0.59 + pdata[(j*4)+2]*0.11;
			pmatrix[j] = grayscaleColor;
		}
	}

	/*print weights*/
	/*sketchCC.clearRect(0, 0, sketchW, sketchH);
	 var nuWeights;
	 for (var i = 0;i < numPatches;i++) {
	 nuWeights = weights[i].map(function(x) {return x*2000+127;});
	 drawData(sketchCC, nuWeights, patchSize, patchSize, false, patchPositions[i][0]-(patchSize/2), patchPositions[i][1]-(patchSize/2));
	 }*/

	// print patches
	/*sketchCC.clearRect(0, 0, sketchW, sketchH);
	 for (var i = 0;i < numPatches;i++) {
	 if ([27,32,44,50].indexOf(i) > -1) {
	 drawData(sketchCC, patches[i], pw, pl, false, patchPositions[i][0]-(pw/2), patchPositions[i][1]-(pl/2));
	 }
	 }*/
	if (patchType == "SVM") {
		if (typeof(webglFi) !== "undefined") {
			responses = getWebGLResponses(patches);
		} else if (typeof(svmFi) !== "undefined"){
			responses = svmFi.getResponses(patches);
		} else {
			throw "SVM-filters do not seem to be initiated properly."
		}
	} else if (patchType == "MOSSE") {
		responses = mosseCalc.getResponses(patches);
	}

	// option to increase sharpness of responses
	if (params.sharpenResponse) {
		for (var i = 0;i < numPatches;i++) {
			for (var j = 0;j < responses[i].length;j++) {
				responses[i][j] = Math.pow(responses[i][j], params.sharpenResponse);
			}
		}
	}

	// print responses
	/*sketchCC.clearRect(0, 0, sketchW, sketchH);
	 var nuWeights;
	 for (var i = 0;i < numPatches;i++) {

	 nuWeights = [];
	 for (var j = 0;j < responses[i].length;j++) {
	 nuWeights.push(responses[i][j]*255);
	 }

	 //if ([27,32,44,50].indexOf(i) > -1) {
	 //	drawData(sketchCC, nuWeights, searchWindow, searchWindow, false, patchPositions[i][0]-((searchWindow-1)/2), patchPositions[i][1]-((searchWindow-1)/2));
	 //}
	 drawData(sketchCC, nuWeights, searchWindow, searchWindow, false, patchPositions[i][0]-((searchWindow-1)/2), patchPositions[i][1]-((searchWindow-1)/2));
	 }*/

	// iterate until convergence or max 10, 20 iterations?:
	var originalPositions = currentPositions;
	var jac;
	var meanshiftVectors = [];

	for (var i = 0; i < varianceSeq.length; i++) {

		// calculate jacobian
		jac = createJacobian(currentParameters, eigenVectors);

		// for debugging
		//var debugMVs = [];
		//

		var opj0, opj1;

		for (var j = 0;j < numPatches;j++) {
			opj0 = originalPositions[j][0]-((searchWindow-1)*scaling/2);
			opj1 = originalPositions[j][1]-((searchWindow-1)*scaling/2);

			// calculate PI x gaussians
			var vpsum = gpopt(searchWindow, currentPositions[j], updatePosition, vecProbs, responses, opj0, opj1, j, varianceSeq[i], scaling);

			// calculate meanshift-vector
			gpopt2(searchWindow, vecpos, updatePosition, vecProbs, vpsum, opj0, opj1, scaling);

			// for debugging
			//var debugMatrixMV = gpopt2(searchWindow, vecpos, updatePosition, vecProbs, vpsum, opj0, opj1);

			// evaluate here whether to increase/decrease stepSize
			/*if (vpsum >= prevCostFunc[j]) {
			 learningRate[j] *= stepParameter;
			 } else {
			 learningRate[j] = 1.0;
			 }
			 prevCostFunc[j] = vpsum;*/

			// compute mean shift vectors
			// extrapolate meanshiftvectors
			/*var msv = [];
			 msv[0] = learningRate[j]*(vecpos[0] - currentPositions[j][0]);
			 msv[1] = learningRate[j]*(vecpos[1] - currentPositions[j][1]);
			 meanshiftVectors[j] = msv;*/
			meanshiftVectors[j] = [vecpos[0] - currentPositions[j][0], vecpos[1] - currentPositions[j][1]];

			//if (isNaN(msv[0]) || isNaN(msv[1])) debugger;

			//for debugging
			//debugMVs[j] = debugMatrixMV;
			//
		}

		// draw meanshiftVector
		/*sketchCC.clearRect(0, 0, sketchW, sketchH);
		 var nuWeights;
		 for (var npidx = 0;npidx < numPatches;npidx++) {
		 nuWeights = debugMVs[npidx].map(function(x) {return x*255*500;});
		 drawData(sketchCC, nuWeights, searchWindow, searchWindow, false, patchPositions[npidx][0]-((searchWindow-1)/2), patchPositions[npidx][1]-((searchWindow-1)/2));
		 }*/

		var meanShiftVector = numeric.rep([numPatches*2, 1],0.0);
		for (var k = 0;k < numPatches;k++) {
			meanShiftVector[k*2][0] = meanshiftVectors[k][0];
			meanShiftVector[(k*2)+1][0] = meanshiftVectors[k][1];
		}

		// compute pdm parameter update
		//var prior = numeric.mul(gaussianPD, PDMVariance);
		var prior = numeric.mul(gaussianPD, varianceSeq[i]);
		if (params.weightPoints) {
			var jtj = numeric.dot(numeric.transpose(jac), numeric.dot(pointWeights, jac));
		} else {
			var jtj = numeric.dot(numeric.transpose(jac), jac);
		}
		var cpMatrix = numeric.rep([numParameters+4, 1],0.0);
		for (var l = 0;l < (numParameters+4);l++) {
			cpMatrix[l][0] = currentParameters[l];
		}
		var priorP = numeric.dot(prior, cpMatrix);
		if (params.weightPoints) {
			var jtv = numeric.dot(numeric.transpose(jac), numeric.dot(pointWeights, meanShiftVector));
		} else {
			var jtv = numeric.dot(numeric.transpose(jac), meanShiftVector);
		}
		var paramUpdateLeft = numeric.add(prior, jtj);
		var paramUpdateRight = numeric.sub(priorP, jtv);
		var paramUpdate = numeric.dot(numeric.inv(paramUpdateLeft), paramUpdateRight);
		//var paramUpdate = numeric.solve(paramUpdateLeft, paramUpdateRight, true);

		var oldPositions = currentPositions;

		// update estimated parameters
		for (var k = 0;k < numParameters+4;k++) {
			currentParameters[k] -= paramUpdate[k];
		}

		// clipping of parameters if they're too high
		var clip;
		for (var k = 0;k < numParameters;k++) {
			clip = Math.abs(3*Math.sqrt(eigenValues[k]));
			if (Math.abs(currentParameters[k+4]) > clip) {
				if (currentParameters[k+4] > 0) {
					currentParameters[k+4] = clip;
				} else {
					currentParameters[k+4] = -clip;
				}
			}

		}

		// update current coordinates
		currentPositions = calculatePositions(currentParameters, true);

		// check if converged
		// calculate norm of parameterdifference
		var positionNorm = 0;
		var pnsq_x, pnsq_y;
		for (var k = 0;k < currentPositions.length;k++) {
			pnsq_x = (currentPositions[k][0]-oldPositions[k][0]);
			pnsq_y = (currentPositions[k][1]-oldPositions[k][1]);
			positionNorm += ((pnsq_x*pnsq_x) + (pnsq_y*pnsq_y));
		}
		//console.log("positionnorm:"+positionNorm);

		// if norm < limit, then break
		if (positionNorm < convergenceLimit) {
			break;
		}

	}

	if (params.constantVelocity) {
		// add current parameter to array of previous parameters
		previousParameters.push(currentParameters.slice());
		previousParameters.splice(0, previousParameters.length == 3 ? 1 : 0);
	}

	// store positions, for checking convergence
	previousPositions.splice(0, previousPositions.length == 10 ? 1 : 0);
	previousPositions.push(currentPositions.slice(0));

	// send an event on each iteration
	var evt = document.createEvent("Event");
	evt.initEvent("clmtrackrIteration", true, true);
	document.dispatchEvent(evt)

	if (this.getConvergence() < 0.5) {
		// we must get a score before we can say we've converged
		if (scoringHistory.length >= 5) {
			if (params.stopOnConvergence) {
				this.stop();
			}

			var evt = document.createEvent("Event");
			evt.initEvent("clmtrackrConverged", true, true);
			document.dispatchEvent(evt)
		}
	}

	// return new points
	return currentPositions;
}

	public getInitialPosition(element, box)
	{
	var translateX, translateY, scaling, rotation;
	if (box) {
		candidate = {x : box[0], y : box[1], width : box[2], height : box[3]};
	} else {
		var det = detectPosition(element);
		if (!det) {
			// if no face found, stop.
			return false;
		}
	}

	if (model.hints && mosseFilter && left_eye_filter && right_eye_filter && nose_filter) {
		var noseFilterWidth = candidate.width * 4.5/10;
		var eyeFilterWidth = candidate.width * 6/10;

		// detect position of eyes and nose via mosse filter
		//
		/*element.pause();

		 var canvasContext = document.getElementById('overlay2').getContext('2d')
		 canvasContext.clearRect(0,0,500,375);
		 canvasContext.strokeRect(candidate.x, candidate.y, candidate.width, candidate.height);*/
		//

		var nose_result = mossef_nose.track(element, Math.round(candidate.x+(candidate.width/2)-(noseFilterWidth/2)), Math.round(candidate.y+candidate.height*(5/8)-(noseFilterWidth/2)), noseFilterWidth, noseFilterWidth, false);
		var right_result = mossef_righteye.track(element, Math.round(candidate.x+(candidate.width*3/4)-(eyeFilterWidth/2)), Math.round(candidate.y+candidate.height*(2/5)-(eyeFilterWidth/2)), eyeFilterWidth, eyeFilterWidth, false);
		var left_result = mossef_lefteye.track(element, Math.round(candidate.x+(candidate.width/4)-(eyeFilterWidth/2)), Math.round(candidate.y+candidate.height*(2/5)-(eyeFilterWidth/2)), eyeFilterWidth, eyeFilterWidth, false);
		right_eye_position[0] = Math.round(candidate.x+(candidate.width*3/4)-(eyeFilterWidth/2))+right_result[0];
		right_eye_position[1] = Math.round(candidate.y+candidate.height*(2/5)-(eyeFilterWidth/2))+right_result[1];
		left_eye_position[0] = Math.round(candidate.x+(candidate.width/4)-(eyeFilterWidth/2))+left_result[0];
		left_eye_position[1] = Math.round(candidate.y+candidate.height*(2/5)-(eyeFilterWidth/2))+left_result[1];
		nose_position[0] = Math.round(candidate.x+(candidate.width/2)-(noseFilterWidth/2))+nose_result[0];
		nose_position[1] = Math.round(candidate.y+candidate.height*(5/8)-(noseFilterWidth/2))+nose_result[1];

		//
		/*canvasContext.strokeRect(Math.round(candidate.x+(candidate.width*3/4)-(eyeFilterWidth/2)), Math.round(candidate.y+candidate.height*(2/5)-(eyeFilterWidth/2)), eyeFilterWidth, eyeFilterWidth);
		 canvasContext.strokeRect(Math.round(candidate.x+(candidate.width/4)-(eyeFilterWidth/2)), Math.round(candidate.y+candidate.height*(2/5)-(eyeFilterWidth/2)), eyeFilterWidth, eyeFilterWidth);
		 //canvasContext.strokeRect(Math.round(candidate.x+(candidate.width/2)-(noseFilterWidth/2)), Math.round(candidate.y+candidate.height*(3/4)-(noseFilterWidth/2)), noseFilterWidth, noseFilterWidth);
		 canvasContext.strokeRect(Math.round(candidate.x+(candidate.width/2)-(noseFilterWidth/2)), Math.round(candidate.y+candidate.height*(5/8)-(noseFilterWidth/2)), noseFilterWidth, noseFilterWidth);

		 canvasContext.fillStyle = "rgb(0,0,250)";
		 canvasContext.beginPath();
		 canvasContext.arc(left_eye_position[0], left_eye_position[1], 3, 0, Math.PI*2, true);
		 canvasContext.closePath();
		 canvasContext.fill();

		 canvasContext.beginPath();
		 canvasContext.arc(right_eye_position[0], right_eye_position[1], 3, 0, Math.PI*2, true);
		 canvasContext.closePath();
		 canvasContext.fill();

		 canvasContext.beginPath();
		 canvasContext.arc(nose_position[0], nose_position[1], 3, 0, Math.PI*2, true);
		 canvasContext.closePath();
		 canvasContext.fill();

		 debugger;
		 element.play()
		 canvasContext.clearRect(0,0,element.width,element.height);*/
		//

		// get eye and nose positions of model
		var lep = model.hints.leftEye;
		var rep = model.hints.rightEye;
		var mep = model.hints.nose;

		// get scaling, rotation, etc. via procrustes analysis
		var procrustes_params = procrustes([left_eye_position, right_eye_position, nose_position], [lep, rep, mep]);
		translateX = procrustes_params[0];
		translateY = procrustes_params[1];
		scaling = procrustes_params[2];
		rotation = procrustes_params[3];

		//element.play();

		//var maxscale = 1.10;
		//if ((scaling*modelHeight)/candidate.height < maxscale*0.7) scaling = (maxscale*0.7*candidate.height)/modelHeight;
		//if ((scaling*modelHeight)/candidate.height > maxscale*1.2) scaling = (maxscale*1.2*candidate.height)/modelHeight;

		/*var smean = [0,0];
		 smean[0] += lep[0];
		 smean[1] += lep[1];
		 smean[0] += rep[0];
		 smean[1] += rep[1];
		 smean[0] += mep[0];
		 smean[1] += mep[1];
		 smean[0] /= 3;
		 smean[1] /= 3;

		 var nulep = [(lep[0]*scaling*Math.cos(-rotation)+lep[1]*scaling*Math.sin(-rotation))+translateX, (lep[0]*scaling*(-Math.sin(-rotation)) + lep[1]*scaling*Math.cos(-rotation))+translateY];
		 var nurep = [(rep[0]*scaling*Math.cos(-rotation)+rep[1]*scaling*Math.sin(-rotation))+translateX, (rep[0]*scaling*(-Math.sin(-rotation)) + rep[1]*scaling*Math.cos(-rotation))+translateY];
		 var numep = [(mep[0]*scaling*Math.cos(-rotation)+mep[1]*scaling*Math.sin(-rotation))+translateX, (mep[0]*scaling*(-Math.sin(-rotation)) + mep[1]*scaling*Math.cos(-rotation))+translateY];

		 canvasContext.fillStyle = "rgb(200,10,100)";
		 canvasContext.beginPath();
		 canvasContext.arc(nulep[0], nulep[1], 3, 0, Math.PI*2, true);
		 canvasContext.closePath();
		 canvasContext.fill();

		 canvasContext.beginPath();
		 canvasContext.arc(nurep[0], nurep[1], 3, 0, Math.PI*2, true);
		 canvasContext.closePath();
		 canvasContext.fill();

		 canvasContext.beginPath();
		 canvasContext.arc(numep[0], numep[1], 3, 0, Math.PI*2, true);
		 canvasContext.closePath();
		 canvasContext.fill();*/

		currentParameters[0] = (scaling*Math.cos(rotation))-1;
		currentParameters[1] = (scaling*Math.sin(rotation));
		currentParameters[2] = translateX;
		currentParameters[3] = translateY;

		//this.draw(document.getElementById('overlay'), currentParameters);

	} else {
		scaling = candidate.width/modelheight;
		//var ccc = document.getElementById('overlay').getContext('2d');
		//ccc.strokeRect(candidate.x,candidate.y,candidate.width,candidate.height);
		translateX = candidate.x-(xmin*scaling)+0.1*candidate.width;
		translateY = candidate.y-(ymin*scaling)+0.25*candidate.height;
		currentParameters[0] = scaling-1;
		currentParameters[2] = translateX;
		currentParameters[3] = translateY;
	}

	currentPositions = calculatePositions(currentParameters, true);

	return [scaling, rotation, translateX, translateY];
}
}