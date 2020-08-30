'use strict';
const HTML_CANVAS = "basimcanvas";
const HTML_RUNNER_MOVEMENTS = "runnermovements";
const HTML_START_BUTTON = "wavestart";
const HTML_LOAD_AGENT = "loadagent";
const HTML_WAVE_SELECT = "waveselect";
const HTML_TICK_COUNT = "tickcount";
const HTML_DEF_LEVEL_SELECT = "deflevelselect";
const HTML_TOGGLE_REPAIR = 'togglerepair'
const HTML_TOGGLE_PAUSE_SL = 'togglepausesl';
const HTML_CURRENT_DEF_FOOD = "currdeffood";
const HTML_TICK_DURATION = "tickduration";
const HTML_TOGGLE_INFINITE_FOOD = "toggleinfinitefood";
const HTML_TOGGLE_LOG_HAMMER_TO_REPAIR = "toggleloghammertorepair";

// RL.JS

var R = {}; // the Recurrent library

(function(global) {
	"use strict";

	// Utility fun
	function assert(condition, message) {
		// from http://stackoverflow.com/questions/15313418/javascript-assert
		if (!condition) {
			message = message || "Assertion failed";
			if (typeof Error !== "undefined") {
				throw new Error(message);
			}
			throw message; // Fallback
		}
	}

	// Random numbers utils
	var return_v = false;
	var v_val = 0.0;
	var gaussRandom = function() {
		if(return_v) {
			return_v = false;
			return v_val;
		}
		var u = 2*Math.random()-1;
		var v = 2*Math.random()-1;
		var r = u*u + v*v;
		if(r == 0 || r > 1) return gaussRandom();
		var c = Math.sqrt(-2*Math.log(r)/r);
		v_val = v*c; // cache this
		return_v = true;
		return u*c;
	}
	var randf = function(a, b) { return Math.random()*(b-a)+a; }
	var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); }
	var randn = function(mu, std){ return mu+gaussRandom()*std; }

	// helper function returns array of zeros of length n
	// and uses typed arrays if available
	var zeros = function(n) {
		if(typeof(n)==='undefined' || isNaN(n)) { return []; }
		if(typeof ArrayBuffer === 'undefined') {
			// lacking browser support
			var arr = new Array(n);
			for(var i=0;i<n;i++) { arr[i] = 0; }
			return arr;
		} else {
			return new Float64Array(n);
		}
	}

	// Mat holds a matrix
	var Mat = function(n,d) {
		// n is number of rows d is number of columns
		this.n = n;
		this.d = d;
		this.w = zeros(n * d);
		this.dw = zeros(n * d);
	}
	Mat.prototype = {
		get: function(row, col) {
			// slow but careful accessor function
			// we want row-major order
			var ix = (this.d * row) + col;
			assert(ix >= 0 && ix < this.w.length);
			return this.w[ix];
		},
		set: function(row, col, v) {
			// slow but careful accessor function
			var ix = (this.d * row) + col;
			assert(ix >= 0 && ix < this.w.length);
			this.w[ix] = v;
		},
		setFrom: function(arr) {
			for(var i=0,n=arr.length;i<n;i++) {
				this.w[i] = arr[i];
			}
		},
		setColumn: function(m, i) {
			for(var q=0,n=m.w.length;q<n;q++) {
				this.w[(this.d * q) + i] = m.w[q];
			}
		},
		toJSON: function() {
			var json = {};
			json['n'] = this.n;
			json['d'] = this.d;
			json['w'] = this.w;
			return json;
		},
		fromJSON: function(json) {
			this.n = json.n;
			this.d = json.d;
			this.w = zeros(this.n * this.d);
			this.dw = zeros(this.n * this.d);
			for(var i=0,n=this.n * this.d;i<n;i++) {
				this.w[i] = json.w[i]; // copy over weights
			}
		}
	}

	var copyMat = function(b) {
		var a = new Mat(b.n, b.d);
		a.setFrom(b.w);
		return a;
	}

	var copyNet = function(net) {
		// nets are (k,v) pairs with k = string key, v = Mat()
		var new_net = {};
		for(var p in net) {
			if(net.hasOwnProperty(p)){
				new_net[p] = copyMat(net[p]);
			}
		}
		return new_net;
	}

	var updateMat = function(m, alpha) {
		// updates in place
		for(var i=0,n=m.n*m.d;i<n;i++) {
			if(m.dw[i] !== 0) {
				m.w[i] += - alpha * m.dw[i];
				m.dw[i] = 0;
			}
		}
	}

	var updateNet = function(net, alpha) {
		for(var p in net) {
			if(net.hasOwnProperty(p)){
				updateMat(net[p], alpha);
			}
		}
	}

	var netToJSON = function(net) {
		var j = {};
		for(var p in net) {
			if(net.hasOwnProperty(p)){
				j[p] = net[p].toJSON();
			}
		}
		return j;
	}
	var netFromJSON = function(j) {
		var net = {};
		for(var p in j) {
			if(j.hasOwnProperty(p)){
				net[p] = new Mat(1,1); // not proud of this
				net[p].fromJSON(j[p]);
			}
		}
		return net;
	}
	var netZeroGrads = function(net) {
		for(var p in net) {
			if(net.hasOwnProperty(p)){
				var mat = net[p];
				gradFillConst(mat, 0);
			}
		}
	}
	var netFlattenGrads = function(net) {
		var n = 0;
		for(var p in net) { if(net.hasOwnProperty(p)){ var mat = net[p]; n += mat.dw.length; } }
		var g = new Mat(n, 1);
		var ix = 0;
		for(var p in net) {
			if(net.hasOwnProperty(p)){
				var mat = net[p];
				for(var i=0,m=mat.dw.length;i<m;i++) {
					g.w[ix] = mat.dw[i];
					ix++;
				}
			}
		}
		return g;
	}

	// return Mat but filled with random numbers from gaussian
	var RandMat = function(n,d,mu,std) {
		var m = new Mat(n, d);
		fillRandn(m,mu,std);
		//fillRand(m,-std,std); // kind of :P
		return m;
	}

	// Mat utils
	// fill matrix with random gaussian numbers
	var fillRandn = function(m, mu, std) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randn(mu, std); } }
	var fillRand = function(m, lo, hi) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randf(lo, hi); } }
	var gradFillConst = function(m, c) { for(var i=0,n=m.dw.length;i<n;i++) { m.dw[i] = c } }

	// Transformer definitions
	var Graph = function(needs_backprop) {
		if(typeof needs_backprop === 'undefined') { needs_backprop = true; }
		this.needs_backprop = needs_backprop;

		// this will store a list of functions that perform backprop,
		// in their forward pass order. So in backprop we will go
		// backwards and evoke each one
		this.backprop = [];
	}
	Graph.prototype = {
		backward: function() {
			for(var i=this.backprop.length-1;i>=0;i--) {
				this.backprop[i](); // tick!
			}
		},
		rowPluck: function(m, ix) {
			// pluck a row of m with index ix and return it as col vector
			assert(ix >= 0 && ix < m.n);
			var d = m.d;
			var out = new Mat(d, 1);
			for(var i=0,n=d;i<n;i++){ out.w[i] = m.w[d * ix + i]; } // copy over the data

			if(this.needs_backprop) {
				var backward = function() {
					for(var i=0,n=d;i<n;i++){ m.dw[d * ix + i] += out.dw[i]; }
				}
				this.backprop.push(backward);
			}
			return out;
		},
		tanh: function(m) {
			// tanh nonlinearity
			var out = new Mat(m.n, m.d);
			var n = m.w.length;
			for(var i=0;i<n;i++) {
				out.w[i] = Math.tanh(m.w[i]);
			}

			if(this.needs_backprop) {
				var backward = function() {
					for(var i=0;i<n;i++) {
						// grad for z = tanh(x) is (1 - z^2)
						var mwi = out.w[i];
						m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
					}
				}
				this.backprop.push(backward);
			}
			return out;
		},
		sigmoid: function(m) {
			// sigmoid nonlinearity
			var out = new Mat(m.n, m.d);
			var n = m.w.length;
			for(var i=0;i<n;i++) {
				out.w[i] = sig(m.w[i]);
			}

			if(this.needs_backprop) {
				var backward = function() {
					for(var i=0;i<n;i++) {
						// grad for z = tanh(x) is (1 - z^2)
						var mwi = out.w[i];
						m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
					}
				}
				this.backprop.push(backward);
			}
			return out;
		},
		relu: function(m) {
			var out = new Mat(m.n, m.d);
			var n = m.w.length;
			for(var i=0;i<n;i++) {
				out.w[i] = Math.max(0, m.w[i]); // relu
			}
			if(this.needs_backprop) {
				var backward = function() {
					for(var i=0;i<n;i++) {
						m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
					}
				}
				this.backprop.push(backward);
			}
			return out;
		},
		mul: function(m1, m2) {
			// multiply matrices m1 * m2
			assert(m1.d === m2.n, 'matmul dimensions misaligned');

			var n = m1.n;
			var d = m2.d;
			var out = new Mat(n,d);
			for(var i=0;i<m1.n;i++) { // loop over rows of m1
				for(var j=0;j<m2.d;j++) { // loop over cols of m2
					var dot = 0.0;
					for(var k=0;k<m1.d;k++) { // dot product loop
						dot += m1.w[m1.d*i+k] * m2.w[m2.d*k+j];
					}
					out.w[d*i+j] = dot;
				}
			}

			if(this.needs_backprop) {
				var backward = function() {
					for(var i=0;i<m1.n;i++) { // loop over rows of m1
						for(var j=0;j<m2.d;j++) { // loop over cols of m2
							for(var k=0;k<m1.d;k++) { // dot product loop
								var b = out.dw[d*i+j];
								m1.dw[m1.d*i+k] += m2.w[m2.d*k+j] * b;
								m2.dw[m2.d*k+j] += m1.w[m1.d*i+k] * b;
							}
						}
					}
				}
				this.backprop.push(backward);
			}
			return out;
		},
		add: function(m1, m2) {
			assert(m1.w.length === m2.w.length);

			var out = new Mat(m1.n, m1.d);
			for(var i=0,n=m1.w.length;i<n;i++) {
				out.w[i] = m1.w[i] + m2.w[i];
			}
			if(this.needs_backprop) {
				var backward = function() {
					for(var i=0,n=m1.w.length;i<n;i++) {
						m1.dw[i] += out.dw[i];
						m2.dw[i] += out.dw[i];
					}
				}
				this.backprop.push(backward);
			}
			return out;
		},
		dot: function(m1, m2) {
			// m1 m2 are both column vectors
			assert(m1.w.length === m2.w.length);
			var out = new Mat(1,1);
			var dot = 0.0;
			for(var i=0,n=m1.w.length;i<n;i++) {
				dot += m1.w[i] * m2.w[i];
			}
			out.w[0] = dot;
			if(this.needs_backprop) {
				var backward = function() {
					for(var i=0,n=m1.w.length;i<n;i++) {
						m1.dw[i] += m2.w[i] * out.dw[0];
						m2.dw[i] += m1.w[i] * out.dw[0];
					}
				}
				this.backprop.push(backward);
			}
			return out;
		},
		eltmul: function(m1, m2) {
			assert(m1.w.length === m2.w.length);

			var out = new Mat(m1.n, m1.d);
			for(var i=0,n=m1.w.length;i<n;i++) {
				out.w[i] = m1.w[i] * m2.w[i];
			}
			if(this.needs_backprop) {
				var backward = function() {
					for(var i=0,n=m1.w.length;i<n;i++) {
						m1.dw[i] += m2.w[i] * out.dw[i];
						m2.dw[i] += m1.w[i] * out.dw[i];
					}
				}
				this.backprop.push(backward);
			}
			return out;
		},
	}

	var softmax = function(m) {
		var out = new Mat(m.n, m.d); // probability volume
		var maxval = -999999;
		for(var i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

		var s = 0.0;
		for(var i=0,n=m.w.length;i<n;i++) {
			out.w[i] = Math.exp(m.w[i] - maxval);
			s += out.w[i];
		}
		for(var i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

		// no backward pass here needed
		// since we will use the computed probabilities outside
		// to set gradients directly on m
		return out;
	}

	var Solver = function() {
		this.decay_rate = 0.999;
		this.smooth_eps = 1e-8;
		this.step_cache = {};
	}
	Solver.prototype = {
		step: function(model, step_size, regc, clipval) {
			// perform parameter update
			var solver_stats = {};
			var num_clipped = 0;
			var num_tot = 0;
			for(var k in model) {
				if(model.hasOwnProperty(k)) {
					var m = model[k]; // mat ref
					if(!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
					var s = this.step_cache[k];
					for(var i=0,n=m.w.length;i<n;i++) {

						// rmsprop adaptive learning rate
						var mdwi = m.dw[i];
						s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

						// gradient clip
						if(mdwi > clipval) {
							mdwi = clipval;
							num_clipped++;
						}
						if(mdwi < -clipval) {
							mdwi = -clipval;
							num_clipped++;
						}
						num_tot++;

						// update (and regularize)
						m.w[i] += - step_size * mdwi / Math.sqrt(s.w[i] + this.smooth_eps) - regc * m.w[i];
						m.dw[i] = 0; // reset gradients for next iteration
					}
				}
			}
			solver_stats['ratio_clipped'] = num_clipped*1.0/num_tot;
			return solver_stats;
		}
	}

	var initLSTM = function(input_size, hidden_sizes, output_size) {
		// hidden size should be a list

		var model = {};
		for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
			var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
			var hidden_size = hidden_sizes[d];

			// gates parameters
			model['Wix'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
			model['Wih'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
			model['bi'+d] = new Mat(hidden_size, 1);
			model['Wfx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
			model['Wfh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
			model['bf'+d] = new Mat(hidden_size, 1);
			model['Wox'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
			model['Woh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
			model['bo'+d] = new Mat(hidden_size, 1);
			// cell write params
			model['Wcx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
			model['Wch'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
			model['bc'+d] = new Mat(hidden_size, 1);
		}
		// decoder params
		model['Whd'] = new RandMat(output_size, hidden_size, 0, 0.08);
		model['bd'] = new Mat(output_size, 1);
		return model;
	}

	var forwardLSTM = function(G, model, hidden_sizes, x, prev) {
		// forward prop for a single tick of LSTM
		// G is graph to append ops to
		// model contains LSTM parameters
		// x is 1D column vector with observation
		// prev is a struct containing hidden and cell
		// from previous iteration

		if(prev == null || typeof prev.h === 'undefined') {
			var hidden_prevs = [];
			var cell_prevs = [];
			for(var d=0;d<hidden_sizes.length;d++) {
				hidden_prevs.push(new R.Mat(hidden_sizes[d],1));
				cell_prevs.push(new R.Mat(hidden_sizes[d],1));
			}
		} else {
			var hidden_prevs = prev.h;
			var cell_prevs = prev.c;
		}

		var hidden = [];
		var cell = [];
		for(var d=0;d<hidden_sizes.length;d++) {

			var input_vector = d === 0 ? x : hidden[d-1];
			var hidden_prev = hidden_prevs[d];
			var cell_prev = cell_prevs[d];

			// input gate
			var h0 = G.mul(model['Wix'+d], input_vector);
			var h1 = G.mul(model['Wih'+d], hidden_prev);
			var input_gate = G.sigmoid(G.add(G.add(h0,h1),model['bi'+d]));

			// forget gate
			var h2 = G.mul(model['Wfx'+d], input_vector);
			var h3 = G.mul(model['Wfh'+d], hidden_prev);
			var forget_gate = G.sigmoid(G.add(G.add(h2, h3),model['bf'+d]));

			// output gate
			var h4 = G.mul(model['Wox'+d], input_vector);
			var h5 = G.mul(model['Woh'+d], hidden_prev);
			var output_gate = G.sigmoid(G.add(G.add(h4, h5),model['bo'+d]));

			// write operation on cells
			var h6 = G.mul(model['Wcx'+d], input_vector);
			var h7 = G.mul(model['Wch'+d], hidden_prev);
			var cell_write = G.tanh(G.add(G.add(h6, h7),model['bc'+d]));

			// compute new cell activation
			var retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
			var write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
			var cell_d = G.add(retain_cell, write_cell); // new cell contents

			// compute hidden state as gated, saturated cell activations
			var hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

			hidden.push(hidden_d);
			cell.push(cell_d);
		}

		// one decoder to outputs at end
		var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]),model['bd']);

		// return cell memory, hidden representation and output
		return {'h':hidden, 'c':cell, 'o' : output};
	}

	var sig = function(x) {
		// helper function for computing sigmoid
		return 1.0/(1+Math.exp(-x));
	}

	var maxi = function(w) {
		// argmax of array w
		var maxv = w[0];
		var maxix = 0;
		for(var i=1,n=w.length;i<n;i++) {
			var v = w[i];
			if(v > maxv) {
				maxix = i;
				maxv = v;
			}
		}
		return maxix;
	}

	var samplei = function(w) {
		// sample argmax from w, assuming w are
		// probabilities that sum to one
		var r = randf(0,1);
		var x = 0.0;
		var i = 0;
		while(true) {
			x += w[i];
			if(x > r) { return i; }
			i++;
		}
		return w.length - 1; // pretty sure we should never get here?
	}

	// various utils
	global.assert = assert;
	global.zeros = zeros;
	global.maxi = maxi;
	global.samplei = samplei;
	global.randi = randi;
	global.randn = randn;
	global.softmax = softmax;
	// classes
	global.Mat = Mat;
	global.RandMat = RandMat;
	global.forwardLSTM = forwardLSTM;
	global.initLSTM = initLSTM;
	// more utils
	global.updateMat = updateMat;
	global.updateNet = updateNet;
	global.copyMat = copyMat;
	global.copyNet = copyNet;
	global.netToJSON = netToJSON;
	global.netFromJSON = netFromJSON;
	global.netZeroGrads = netZeroGrads;
	global.netFlattenGrads = netFlattenGrads;
	// optimization
	global.Solver = Solver;
	global.Graph = Graph;
})(R);

// END OF RECURRENTJS

var RL = {};
(function(global) {
	"use strict";

// syntactic sugar function for getting default parameter values
	var getopt = function(opt, field_name, default_value) {
		if(typeof opt === 'undefined') { return default_value; }
		return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
	}

	var zeros = R.zeros; // inherit these
	var assert = R.assert;
	var randi = R.randi;
	var randf = R.randf;

	var setConst = function(arr, c) {
		for(var i=0,n=arr.length;i<n;i++) {
			arr[i] = c;
		}
	}

	var sampleWeighted = function(p) {
		var r = Math.random();
		var c = 0.0;
		for(var i=0,n=p.length;i<n;i++) {
			c += p[i];
			if(c >= r) { return i; }
		}
		assert(false, 'wtf');
	}

// ------
// AGENTS
// ------

// DPAgent performs Value Iteration
// - can also be used for Policy Iteration if you really wanted to
// - requires model of the environment :(
// - does not learn from experience :(
// - assumes finite MDP :(
	var DPAgent = function(env, opt) {
		this.V = null; // state value function
		this.P = null; // policy distribution \pi(s,a)
		this.env = env; // store pointer to environment
		this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
		this.reset();
	}
	DPAgent.prototype = {
		reset: function() {
			// reset the agent's policy and value function
			this.ns = this.env.getNumStates();
			this.na = this.env.getMaxNumActions();
			this.V = zeros(this.ns);
			this.P = zeros(this.ns * this.na);
			// initialize uniform random policy
			for(var s=0;s<this.ns;s++) {
				var poss = this.env.allowedActions(s);
				for(var i=0,n=poss.length;i<n;i++) {
					this.P[poss[i]*this.ns+s] = 1.0 / poss.length;
				}
			}
		},
		act: function(s) {
			// behave according to the learned policy
			var poss = this.env.allowedActions(s);
			var ps = [];
			for(var i=0,n=poss.length;i<n;i++) {
				var a = poss[i];
				var prob = this.P[a*this.ns+s];
				ps.push(prob);
			}
			var maxi = sampleWeighted(ps);
			return poss[maxi];
		},
		learn: function() {
			// perform a single round of value iteration
			self.evaluatePolicy(); // writes this.V
			self.updatePolicy(); // writes this.P
		},
		evaluatePolicy: function() {
			// perform a synchronous update of the value function
			var Vnew = zeros(this.ns);
			for(var s=0;s<this.ns;s++) {
				// integrate over actions in a stochastic policy
				// note that we assume that policy probability mass over allowed actions sums to one
				var v = 0.0;
				var poss = this.env.allowedActions(s);
				for(var i=0,n=poss.length;i<n;i++) {
					var a = poss[i];
					var prob = this.P[a*this.ns+s]; // probability of taking action under policy
					if(prob === 0) { continue; } // no contribution, skip for speed
					var ns = this.env.nextStateDistribution(s,a);
					var rs = this.env.reward(s,a,ns); // reward for s->a->ns transition
					v += prob * (rs + this.gamma * this.V[ns]);
				}
				Vnew[s] = v;
			}
			this.V = Vnew; // swap
		},
		updatePolicy: function() {
			// update policy to be greedy w.r.t. learned Value function
			for(var s=0;s<this.ns;s++) {
				var poss = this.env.allowedActions(s);
				// compute value of taking each allowed action
				var vmax, nmax;
				var vs = [];
				for(var i=0,n=poss.length;i<n;i++) {
					var a = poss[i];
					var ns = this.env.nextStateDistribution(s,a);
					var rs = this.env.reward(s,a,ns);
					var v = rs + this.gamma * this.V[ns];
					vs.push(v);
					if(i === 0 || v > vmax) { vmax = v; nmax = 1; }
					else if(v === vmax) { nmax += 1; }
				}
				// update policy smoothly across all argmaxy actions
				for(var i=0,n=poss.length;i<n;i++) {
					var a = poss[i];
					this.P[a*this.ns+s] = (vs[i] === vmax) ? 1.0/nmax : 0.0;
				}
			}
		},
	}

// QAgent uses TD (Q-Learning, SARSA)
// - does not require environment model :)
// - learns from experience :)
	var TDAgent = function(env, opt) {
		this.update = getopt(opt, 'update', 'qlearn'); // qlearn | sarsa
		this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
		this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
		this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

		// class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
		this.smooth_policy_update = getopt(opt, 'smooth_policy_update', false);
		this.beta = getopt(opt, 'beta', 0.01); // learning rate for policy, if smooth updates are on

		// eligibility traces
		this.lambda = getopt(opt, 'lambda', 0); // eligibility trace decay. 0 = no eligibility traces used
		this.replacing_traces = getopt(opt, 'replacing_traces', true);

		// optional optimistic initial values
		this.q_init_val = getopt(opt, 'q_init_val', 0);

		this.planN = getopt(opt, 'planN', 0); // number of planning steps per learning iteration (0 = no planning)

		this.Q = null; // state action value function
		this.P = null; // policy distribution \pi(s,a)
		this.e = null; // eligibility trace
		this.env_model_s = null;; // environment model (s,a) -> (s',r)
		this.env_model_r = null;; // environment model (s,a) -> (s',r)
		this.env = env; // store pointer to environment
		this.reset();
	}
	TDAgent.prototype = {
		reset: function(){
			// reset the agent's policy and value function
			this.ns = this.env.getNumStates();
			this.na = this.env.getMaxNumActions();
			this.Q = zeros(this.ns * this.na);
			if(this.q_init_val !== 0) { setConst(this.Q, this.q_init_val); }
			this.P = zeros(this.ns * this.na);
			this.e = zeros(this.ns * this.na);

			// model/planning vars
			this.env_model_s = zeros(this.ns * this.na);
			setConst(this.env_model_s, -1); // init to -1 so we can test if we saw the state before
			this.env_model_r = zeros(this.ns * this.na);
			this.sa_seen = [];
			this.pq = zeros(this.ns * this.na);

			// initialize uniform random policy
			for(var s=0;s<this.ns;s++) {
				var poss = this.env.allowedActions(s);
				for(var i=0,n=poss.length;i<n;i++) {
					this.P[poss[i]*this.ns+s] = 1.0 / poss.length;
				}
			}
			// agent memory, needed for streaming updates
			// (s0,a0,r0,s1,a1,r1,...)
			this.r0 = null;
			this.s0 = null;
			this.s1 = null;
			this.a0 = null;
			this.a1 = null;
		},
		resetEpisode: function() {
			// an episode finished
		},
		act: function(s){
			// act according to epsilon greedy policy
			var poss = this.env.allowedActions(s);
			var probs = [];
			for(var i=0,n=poss.length;i<n;i++) {
				probs.push(this.P[poss[i]*this.ns+s]);
			}
			// epsilon greedy policy
			if(Math.random() < this.epsilon) {
				var a = poss[randi(0,poss.length)]; // random available action
				this.explored = true;
			} else {
				var a = poss[sampleWeighted(probs)];
				this.explored = false;
			}
			// shift state memory
			this.s0 = this.s1;
			this.a0 = this.a1;
			this.s1 = s;
			this.a1 = a;
			return a;
		},
		learn: function(r1){
			// takes reward for previous action, which came from a call to act()
			if(!(this.r0 == null)) {
				this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda);
				if(this.planN > 0) {
					this.updateModel(this.s0, this.a0, this.r0, this.s1);
					this.plan();
				}
			}
			this.r0 = r1; // store this for next update
		},
		updateModel: function(s0, a0, r0, s1) {
			// transition (s0,a0) -> (r0,s1) was observed. Update environment model
			var sa = a0 * this.ns + s0;
			if(this.env_model_s[sa] === -1) {
				// first time we see this state action
				this.sa_seen.push(a0 * this.ns + s0); // add as seen state
			}
			this.env_model_s[sa] = s1;
			this.env_model_r[sa] = r0;
		},
		plan: function() {

			// order the states based on current priority queue information
			var spq = [];
			for(var i=0,n=this.sa_seen.length;i<n;i++) {
				var sa = this.sa_seen[i];
				var sap = this.pq[sa];
				if(sap > 1e-5) { // gain a bit of efficiency
					spq.push({sa:sa, p:sap});
				}
			}
			spq.sort(function(a,b){ return a.p < b.p ? 1 : -1});

			// perform the updates
			var nsteps = Math.min(this.planN, spq.length);
			for(var k=0;k<nsteps;k++) {
				// random exploration
				//var i = randi(0, this.sa_seen.length); // pick random prev seen state action
				//var s0a0 = this.sa_seen[i];
				var s0a0 = spq[k].sa;
				this.pq[s0a0] = 0; // erase priority, since we're backing up this state
				var s0 = s0a0 % this.ns;
				var a0 = Math.floor(s0a0 / this.ns);
				var r0 = this.env_model_r[s0a0];
				var s1 = this.env_model_s[s0a0];
				var a1 = -1; // not used for Q learning
				if(this.update === 'sarsa') {
					// generate random action?...
					var poss = this.env.allowedActions(s1);
					var a1 = poss[randi(0,poss.length)];
				}
				this.learnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
			}
		},
		learnFromTuple: function(s0, a0, r0, s1, a1, lambda) {
			var sa = a0 * this.ns + s0;

			// calculate the target for Q(s,a)
			if(this.update === 'qlearn') {
				// Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
				var poss = this.env.allowedActions(s1);
				var qmax = 0;
				for(var i=0,n=poss.length;i<n;i++) {
					var s1a = poss[i] * this.ns + s1;
					var qval = this.Q[s1a];
					if(i === 0 || qval > qmax) { qmax = qval; }
				}
				var target = r0 + this.gamma * qmax;
			} else if(this.update === 'sarsa') {
				// SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
				var s1a1 = a1 * this.ns + s1;
				var target = r0 + this.gamma * this.Q[s1a1];
			}

			if(lambda > 0) {
				// perform an eligibility trace update
				if(this.replacing_traces) {
					this.e[sa] = 1;
				} else {
					this.e[sa] += 1;
				}
				var edecay = lambda * this.gamma;
				var state_update = zeros(this.ns);
				for(var s=0;s<this.ns;s++) {
					var poss = this.env.allowedActions(s);
					for(var i=0;i<poss.length;i++) {
						var a = poss[i];
						var saloop = a * this.ns + s;
						var esa = this.e[saloop];
						var update = this.alpha * esa * (target - this.Q[saloop]);
						this.Q[saloop] += update;
						this.updatePriority(s, a, update);
						this.e[saloop] *= edecay;
						var u = Math.abs(update);
						if(u > state_update[s]) { state_update[s] = u; }
					}
				}
				for(var s=0;s<this.ns;s++) {
					if(state_update[s] > 1e-5) { // save efficiency here
						this.updatePolicy(s);
					}
				}
				if(this.explored && this.update === 'qlearn') {
					// have to wipe the trace since q learning is off-policy :(
					this.e = zeros(this.ns * this.na);
				}
			} else {
				// simpler and faster update without eligibility trace
				// update Q[sa] towards it with some step size
				var update = this.alpha * (target - this.Q[sa]);
				this.Q[sa] += update;
				this.updatePriority(s0, a0, update);
				// update the policy to reflect the change (if appropriate)
				this.updatePolicy(s0);
			}
		},
		updatePriority: function(s,a,u) {
			// used in planning. Invoked when Q[sa] += update
			// we should find all states that lead to (s,a) and upgrade their priority
			// of being update in the next planning step
			u = Math.abs(u);
			if(u < 1e-5) { return; } // for efficiency skip small updates
			if(this.planN === 0) { return; } // there is no planning to be done, skip.
			for(var si=0;si<this.ns;si++) {
				// note we are also iterating over impossible actions at all states,
				// but this should be okay because their env_model_s should simply be -1
				// as initialized, so they will never be predicted to point to any state
				// because they will never be observed, and hence never be added to the model
				for(var ai=0;ai<this.na;ai++) {
					var siai = ai * this.ns + si;
					if(this.env_model_s[siai] === s) {
						// this state leads to s, add it to priority queue
						this.pq[siai] += u;
					}
				}
			}
		},
		updatePolicy: function(s) {
			var poss = this.env.allowedActions(s);
			// set policy at s to be the action that achieves max_a Q(s,a)
			// first find the maxy Q values
			var qmax, nmax;
			var qs = [];
			for(var i=0,n=poss.length;i<n;i++) {
				var a = poss[i];
				var qval = this.Q[a*this.ns+s];
				qs.push(qval);
				if(i === 0 || qval > qmax) { qmax = qval; nmax = 1; }
				else if(qval === qmax) { nmax += 1; }
			}
			// now update the policy smoothly towards the argmaxy actions
			var psum = 0.0;
			for(var i=0,n=poss.length;i<n;i++) {
				var a = poss[i];
				var target = (qs[i] === qmax) ? 1.0/nmax : 0.0;
				var ix = a*this.ns+s;
				if(this.smooth_policy_update) {
					// slightly hacky :p
					this.P[ix] += this.beta * (target - this.P[ix]);
					psum += this.P[ix];
				} else {
					// set hard target
					this.P[ix] = target;
				}
			}
			if(this.smooth_policy_update) {
				// renomalize P if we're using smooth policy updates
				for(var i=0,n=poss.length;i<n;i++) {
					var a = poss[i];
					this.P[a*this.ns+s] /= psum;
				}
			}
		}
	}


	var DQNAgent = function(env, opt) {
		this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
		this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
		this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

		this.experience_add_every = getopt(opt, 'experience_add_every', 25); // number of time steps before we add another experience to replay memory
		this.experience_size = getopt(opt, 'experience_size', 5000); // size of experience replay
		this.learning_steps_per_iteration = getopt(opt, 'learning_steps_per_iteration', 10);
		this.tderror_clamp = getopt(opt, 'tderror_clamp', 1.0);

		this.num_hidden_units =  getopt(opt, 'num_hidden_units', 100);

		this.env = env;
		this.reset();
	}
	DQNAgent.prototype = {
		reset: function() {
			this.nh = this.num_hidden_units; // number of hidden units
			this.ns = this.env.getNumStates();
			this.na = this.env.getMaxNumActions();

			// nets are hardcoded for now as key (str) -> Mat
			// not proud of this. better solution is to have a whole Net object
			// on top of Mats, but for now sticking with this
			this.net = {};
			this.net.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
			this.net.b1 = new R.Mat(this.nh, 1, 0, 0.01);
			this.net.W2 = new R.RandMat(this.na, this.nh, 0, 0.01);
			this.net.b2 = new R.Mat(this.na, 1, 0, 0.01);

			this.exp = []; // experience
			this.expi = 0; // where to insert

			this.t = 0;

			this.r0 = null;
			this.s0 = null;
			this.s1 = null;
			this.a0 = null;
			this.a1 = null;

			this.tderror = 0; // for visualization only...
		},
		toJSON: function() {
			// save function
			var j = {};
			j.nh = this.nh;
			j.ns = this.ns;
			j.na = this.na;
			j.net = R.netToJSON(this.net);
			return j;
		},
		fromJSON: function(j) {
			// load function
			this.nh = j.nh;
			this.ns = j.ns;
			this.na = j.na;
			this.net = R.netFromJSON(j.net);
		},
		forwardQ: function(net, s, needs_backprop) {
			var G = new R.Graph(needs_backprop);
			var a1mat = G.add(G.mul(net.W1, s), net.b1);
			var h1mat = G.tanh(a1mat);
			var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
			this.lastG = G; // back this up. Kind of hacky isn't it
			return a2mat;
		},
		act: function(slist) {
			// convert to a Mat column vector
			var s = new R.Mat(this.ns, 1);
			s.setFrom(slist);

			// epsilon greedy policy
			if(Math.random() < this.epsilon) {
				var a = randi(0, this.na);
			} else {
				// greedy wrt Q function
				var amat = this.forwardQ(this.net, s, false);
				var a = R.maxi(amat.w); // returns index of argmax action
			}

			// shift state memory
			this.s0 = this.s1;
			this.a0 = this.a1;
			this.s1 = s;
			this.a1 = a;

			return a;
		},
		learn: function(r1) {
			// perform an update on Q function
			if(!(this.r0 == null) && this.alpha > 0) {

				// learn from this tuple to get a sense of how "surprising" it is to the agent
				var tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1);
				this.tderror = tderror; // a measure of surprise

				// decide if we should keep this experience in the replay
				if(this.t % this.experience_add_every === 0) {
					this.exp[this.expi] = [this.s0, this.a0, this.r0, this.s1, this.a1];
					this.expi += 1;
					if(this.expi > this.experience_size) { this.expi = 0; } // roll over when we run out
				}
				this.t += 1;

				// sample some additional experience from replay memory and learn from it
				for(var k=0;k<this.learning_steps_per_iteration;k++) {
					var ri = randi(0, this.exp.length); // todo: priority sweeps?
					var e = this.exp[ri];
					this.learnFromTuple(e[0], e[1], e[2], e[3], e[4])
				}
			}
			this.r0 = r1; // store for next update
		},
		learnFromTuple: function(s0, a0, r0, s1, a1) {
			// want: Q(s,a) = r + gamma * max_a' Q(s',a')

			// compute the target Q value
			var tmat = this.forwardQ(this.net, s1, false);
			var qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)];

			// now predict
			var pred = this.forwardQ(this.net, s0, true);

			var tderror = pred.w[a0] - qmax;
			var clamp = this.tderror_clamp;
			if(Math.abs(tderror) > clamp) {  // huber loss to robustify
				if(tderror > clamp) tderror = clamp;
				if(tderror < -clamp) tderror = -clamp;
			}
			pred.dw[a0] = tderror;
			this.lastG.backward(); // compute gradients on net params

			// update net
			R.updateNet(this.net, this.alpha);
			return tderror;
		}
	}

// buggy implementation, doesnt work...
	var SimpleReinforceAgent = function(env, opt) {
		this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
		this.epsilon = getopt(opt, 'epsilon', 0.75); // for epsilon-greedy policy
		this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
		this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
		this.env = env;
		this.reset();
	}
	SimpleReinforceAgent.prototype = {
		reset: function() {
			this.ns = this.env.getNumStates();
			this.na = this.env.getMaxNumActions();
			this.nh = 100; // number of hidden units
			this.nhb = 100; // and also in the baseline lstm

			this.actorNet = {};
			this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
			this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
			this.actorNet.W2 = new R.RandMat(this.na, this.nh, 0, 0.1);
			this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
			this.actorOutputs = [];
			this.actorGraphs = [];
			this.actorActions = []; // sampled ones

			this.rewardHistory = [];

			this.baselineNet = {};
			this.baselineNet.W1 = new R.RandMat(this.nhb, this.ns, 0, 0.01);
			this.baselineNet.b1 = new R.Mat(this.nhb, 1, 0, 0.01);
			this.baselineNet.W2 = new R.RandMat(this.na, this.nhb, 0, 0.01);
			this.baselineNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
			this.baselineOutputs = [];
			this.baselineGraphs = [];

			this.t = 0;
		},
		forwardActor: function(s, needs_backprop) {
			var net = this.actorNet;
			var G = new R.Graph(needs_backprop);
			var a1mat = G.add(G.mul(net.W1, s), net.b1);
			var h1mat = G.tanh(a1mat);
			var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
			return {'a':a2mat, 'G':G}
		},
		forwardValue: function(s, needs_backprop) {
			var net = this.baselineNet;
			var G = new R.Graph(needs_backprop);
			var a1mat = G.add(G.mul(net.W1, s), net.b1);
			var h1mat = G.tanh(a1mat);
			var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
			return {'a':a2mat, 'G':G}
		},
		act: function(slist) {
			// convert to a Mat column vector
			var s = new R.Mat(this.ns, 1);
			s.setFrom(slist);

			// forward the actor to get action output
			var ans = this.forwardActor(s, true);
			var amat = ans.a;
			var ag = ans.G;
			this.actorOutputs.push(amat);
			this.actorGraphs.push(ag);

			// forward the baseline estimator
			var ans = this.forwardValue(s, true);
			var vmat = ans.a;
			var vg = ans.G;
			this.baselineOutputs.push(vmat);
			this.baselineGraphs.push(vg);

			// sample action from the stochastic gaussian policy
			var a = R.copyMat(amat);
			var gaussVar = 0.02;
			a.w[0] = R.randn(0, gaussVar);
			a.w[1] = R.randn(0, gaussVar);

			this.actorActions.push(a);

			// shift state memory
			this.s0 = this.s1;
			this.a0 = this.a1;
			this.s1 = s;
			this.a1 = a;

			return a;
		},
		learn: function(r1) {
			// perform an update on Q function
			this.rewardHistory.push(r1);
			var n = this.rewardHistory.length;
			var baselineMSE = 0.0;
			var nup = 100; // what chunk of experience to take
			var nuse = 80; // what chunk to update from
			if(n >= nup) {
				// lets learn and flush
				// first: compute the sample values at all points
				var vs = [];
				for(var t=0;t<nuse;t++) {
					var mul = 1;
					// compute the actual discounted reward for this time step
					var V = 0;
					for(var t2=t;t2<n;t2++) {
						V += mul * this.rewardHistory[t2];
						mul *= this.gamma;
						if(mul < 1e-5) { break; } // efficiency savings
					}
					// get the predicted baseline at this time step
					var b = this.baselineOutputs[t].w[0];
					for(var i=0;i<this.na;i++) {
						// [the action delta] * [the desirebility]
						var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
						if(update > 0.1) { update = 0.1; }
						if(update < -0.1) { update = -0.1; }
						this.actorOutputs[t].dw[i] += update;
					}
					var update = - (V - b);
					if(update > 0.1) { update = 0.1; }
					if(update < 0.1) { update = -0.1; }
					this.baselineOutputs[t].dw[0] += update;
					baselineMSE += (V - b) * (V - b);
					vs.push(V);
				}
				baselineMSE /= nuse;
				// backprop all the things
				for(var t=0;t<nuse;t++) {
					this.actorGraphs[t].backward();
					this.baselineGraphs[t].backward();
				}
				R.updateNet(this.actorNet, this.alpha); // update actor network
				R.updateNet(this.baselineNet, this.beta); // update baseline network

				// flush
				this.actorOutputs = [];
				this.rewardHistory = [];
				this.actorActions = [];
				this.baselineOutputs = [];
				this.actorGraphs = [];
				this.baselineGraphs = [];

				this.tderror = baselineMSE;
			}
			this.t += 1;
			this.r0 = r1; // store for next update
		},
	}

// buggy implementation as well, doesn't work
	var RecurrentReinforceAgent = function(env, opt) {
		this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
		this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
		this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
		this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
		this.env = env;
		this.reset();
	}
	RecurrentReinforceAgent.prototype = {
		reset: function() {
			this.ns = this.env.getNumStates();
			this.na = this.env.getMaxNumActions();
			this.nh = 40; // number of hidden units
			this.nhb = 40; // and also in the baseline lstm

			this.actorLSTM = R.initLSTM(this.ns, [this.nh], this.na);
			this.actorG = new R.Graph();
			this.actorPrev = null;
			this.actorOutputs = [];
			this.rewardHistory = [];
			this.actorActions = [];

			this.baselineLSTM = R.initLSTM(this.ns, [this.nhb], 1);
			this.baselineG = new R.Graph();
			this.baselinePrev = null;
			this.baselineOutputs = [];

			this.t = 0;

			this.r0 = null;
			this.s0 = null;
			this.s1 = null;
			this.a0 = null;
			this.a1 = null;
		},
		act: function(slist) {
			// convert to a Mat column vector
			var s = new R.Mat(this.ns, 1);
			s.setFrom(slist);

			// forward the LSTM to get action distribution
			var actorNext = R.forwardLSTM(this.actorG, this.actorLSTM, [this.nh], s, this.actorPrev);
			this.actorPrev = actorNext;
			var amat = actorNext.o;
			this.actorOutputs.push(amat);

			// forward the baseline LSTM
			var baselineNext = R.forwardLSTM(this.baselineG, this.baselineLSTM, [this.nhb], s, this.baselinePrev);
			this.baselinePrev = baselineNext;
			this.baselineOutputs.push(baselineNext.o);

			// sample action from actor policy
			var gaussVar = 0.05;
			var a = R.copyMat(amat);
			for(var i=0,n=a.w.length;i<n;i++) {
				a.w[0] += R.randn(0, gaussVar);
				a.w[1] += R.randn(0, gaussVar);
			}
			this.actorActions.push(a);

			// shift state memory
			this.s0 = this.s1;
			this.a0 = this.a1;
			this.s1 = s;
			this.a1 = a;
			return a;
		},
		learn: function(r1) {
			// perform an update on Q function
			this.rewardHistory.push(r1);
			var n = this.rewardHistory.length;
			var baselineMSE = 0.0;
			var nup = 100; // what chunk of experience to take
			var nuse = 80; // what chunk to also update
			if(n >= nup) {
				// lets learn and flush
				// first: compute the sample values at all points
				var vs = [];
				for(var t=0;t<nuse;t++) {
					var mul = 1;
					var V = 0;
					for(var t2=t;t2<n;t2++) {
						V += mul * this.rewardHistory[t2];
						mul *= this.gamma;
						if(mul < 1e-5) { break; } // efficiency savings
					}
					var b = this.baselineOutputs[t].w[0];
					// todo: take out the constants etc.
					for(var i=0;i<this.na;i++) {
						// [the action delta] * [the desirebility]
						var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
						if(update > 0.1) { update = 0.1; }
						if(update < -0.1) { update = -0.1; }
						this.actorOutputs[t].dw[i] += update;
					}
					var update = - (V - b);
					if(update > 0.1) { update = 0.1; }
					if(update < 0.1) { update = -0.1; }
					this.baselineOutputs[t].dw[0] += update;
					baselineMSE += (V-b)*(V-b);
					vs.push(V);
				}
				baselineMSE /= nuse;
				this.actorG.backward(); // update params! woohoo!
				this.baselineG.backward();
				R.updateNet(this.actorLSTM, this.alpha); // update actor network
				R.updateNet(this.baselineLSTM, this.beta); // update baseline network

				// flush
				this.actorG = new R.Graph();
				this.actorPrev = null;
				this.actorOutputs = [];
				this.rewardHistory = [];
				this.actorActions = [];

				this.baselineG = new R.Graph();
				this.baselinePrev = null;
				this.baselineOutputs = [];

				this.tderror = baselineMSE;
			}
			this.t += 1;
			this.r0 = r1; // store for next update
		},
	}

// Currently buggy implementation, doesnt work
	var DeterministPG = function(env, opt) {
		this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
		this.epsilon = getopt(opt, 'epsilon', 0.5); // for epsilon-greedy policy
		this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
		this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
		this.env = env;
		this.reset();
	}
	DeterministPG.prototype = {
		reset: function() {
			this.ns = this.env.getNumStates();
			this.na = this.env.getMaxNumActions();
			this.nh = 100; // number of hidden units

			// actor
			this.actorNet = {};
			this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
			this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
			this.actorNet.W2 = new R.RandMat(this.na, this.ns, 0, 0.1);
			this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
			this.ntheta = this.na*this.ns+this.na; // number of params in actor

			// critic
			this.criticw = new R.RandMat(1, this.ntheta, 0, 0.01); // row vector

			this.r0 = null;
			this.s0 = null;
			this.s1 = null;
			this.a0 = null;
			this.a1 = null;
			this.t = 0;
		},
		forwardActor: function(s, needs_backprop) {
			var net = this.actorNet;
			var G = new R.Graph(needs_backprop);
			var a1mat = G.add(G.mul(net.W1, s), net.b1);
			var h1mat = G.tanh(a1mat);
			var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
			return {'a':a2mat, 'G':G}
		},
		act: function(slist) {
			// convert to a Mat column vector
			var s = new R.Mat(this.ns, 1);
			s.setFrom(slist);

			// forward the actor to get action output
			var ans = this.forwardActor(s, false);
			var amat = ans.a;
			var ag = ans.G;

			// sample action from the stochastic gaussian policy
			var a = R.copyMat(amat);
			if(Math.random() < this.epsilon) {
				var gaussVar = 0.02;
				a.w[0] = R.randn(0, gaussVar);
				a.w[1] = R.randn(0, gaussVar);
			}
			var clamp = 0.25;
			if(a.w[0] > clamp) a.w[0] = clamp;
			if(a.w[0] < -clamp) a.w[0] = -clamp;
			if(a.w[1] > clamp) a.w[1] = clamp;
			if(a.w[1] < -clamp) a.w[1] = -clamp;

			// shift state memory
			this.s0 = this.s1;
			this.a0 = this.a1;
			this.s1 = s;
			this.a1 = a;

			return a;
		},
		utilJacobianAt: function(s) {
			var ujacobian = new R.Mat(this.ntheta, this.na);
			for(var a=0;a<this.na;a++) {
				R.netZeroGrads(this.actorNet);
				var ag = this.forwardActor(this.s0, true);
				ag.a.dw[a] = 1.0;
				ag.G.backward();
				var gflat = R.netFlattenGrads(this.actorNet);
				ujacobian.setColumn(gflat,a);
			}
			return ujacobian;
		},
		learn: function(r1) {
			// perform an update on Q function
			//this.rewardHistory.push(r1);
			if(!(this.r0 == null)) {
				var Gtmp = new R.Graph(false);
				// dpg update:
				// first compute the features psi:
				// the jacobian matrix of the actor for s
				var ujacobian0 = this.utilJacobianAt(this.s0);
				// now form the features \psi(s,a)
				var psi_sa0 = Gtmp.mul(ujacobian0, this.a0); // should be [this.ntheta x 1] "feature" vector
				var qw0 = Gtmp.mul(this.criticw, psi_sa0); // 1x1
				// now do the same thing because we need \psi(s_{t+1}, \mu\_\theta(s\_t{t+1}))
				var ujacobian1 = this.utilJacobianAt(this.s1);
				var ag = this.forwardActor(this.s1, false);
				var psi_sa1 = Gtmp.mul(ujacobian1, ag.a);
				var qw1 = Gtmp.mul(this.criticw, psi_sa1); // 1x1
				// get the td error finally
				var tderror = this.r0 + this.gamma * qw1.w[0] - qw0.w[0]; // lol
				if(tderror > 0.5) tderror = 0.5; // clamp
				if(tderror < -0.5) tderror = -0.5;
				this.tderror = tderror;

				// update actor policy with natural gradient
				var net = this.actorNet;
				var ix = 0;
				for(var p in net) {
					var mat = net[p];
					if(net.hasOwnProperty(p)){
						for(var i=0,n=mat.w.length;i<n;i++) {
							mat.w[i] += this.alpha * this.criticw.w[ix]; // natural gradient update
							ix+=1;
						}
					}
				}
				// update the critic parameters too
				for(var i=0;i<this.ntheta;i++) {
					var update = this.beta * tderror * psi_sa0.w[i];
					this.criticw.w[i] += update;
				}
			}
			this.r0 = r1; // store for next update
		},
	}

// exports
	global.DPAgent = DPAgent;
	global.TDAgent = TDAgent;
	global.DQNAgent = DQNAgent;
//global.SimpleReinforceAgent = SimpleReinforceAgent;
//global.RecurrentReinforceAgent = RecurrentReinforceAgent;
//global.DeterministPG = DeterministPG;
})(RL);

// SOME RL VARS

var foodList;

var Wave1TestAgent = function() {
	this.num_states = 2;
	this.actions = [];
	for (let i = 0; i < 3; i++) {
		this.actions.push(i);
	}
	this.action = 0;
	this.brain = null;
}
Wave1TestAgent.prototype = {
	getNumStates: function() {
		return this.num_states;
	},
	getMaxNumActions: function() {
		return this.actions.length;
	},
	forward: function() {
		var input_array = new Array(this.num_states);
		input_array[0] = plDefX;
		input_array[1] = plDefY;
		this.action = this.brain.act(input_array);
		this.performAction(this.action);
	},
	backward: function() {
		var reward = (plDefY === 25) ? 1 : 0;
		this.last_reward = reward;
		this.brain.learn(reward);
	},
	performAction: function(a) {
		let dx = 0;
		let dy = 0;

		if (a === 1) {
			dy = 1;
		} else if (a === 2) {
			dy = -1;
		}

		plDefPathfind(plDefX + dx, plDefY + dy);
	}
}

var RunnerWave1Agent = function() {
	this.num_states = 124;

	this.actions = [];
	for (let i = 0; i < 403; i++) {
		this.actions.push(i);
	}

	this.action = 0;

	this.brain = null;
}
RunnerWave1Agent.prototype = {
	getNumStates: function() {
		return this.num_states;
	},
	getMaxNumActions: function() {
		return this.actions.length; // drop up to 5 things AND: logs, hammer, repair, move (9), pick t, pick c, pick w
	},
	forward: function() {
		var input_array = new Array(this.num_states);

		input_array[0] = numTofu;
		input_array[1] = numCrackers;
		input_array[2] = numWorms;

		if (currDefFood === "t") {
			input_array[3] = 1;
		} else {
			input_array[3] = (currDefFood === "c") ? 2 : 3;
		}

		input_array[4] = numLogs;
		input_array[5] = hasHammer ? 1 : 0;
		input_array[6] = eastTrapState;
		input_array[7] = westTrapState;
		input_array[8] = northwestLogsState ? 1 : 0;
		input_array[9] = southeastLogsState ? 1 : 0;
		input_array[10] = hammerState ? 1 : 0;

		if (pickingUpFood === "t") {
			input_array[11] = 1;
		} else if (pickingUpFood === "c") {
			input_array[11] = 2;
		} else {
			input_array[11] = (pickingUpFood === "w") ? 3 : 4;
		}

		input_array[12] = pickingUpLogs ? 1 : 0;
		input_array[13] = pickingUpHammer ? 1 : 0;
		input_array[14] = repairTicksRemaining;
		input_array[15] = plDefX;
		input_array[16] = plDefY;
		input_array[17] = plDefStandStillCounter;
		input_array[18] = 0 //baTickCounter % 50;
		input_array[19] = baRunnersAlive;
		input_array[20] = baRunnersKilled;
		input_array[21] = baTotalRunners;
		input_array[22] = baMaxRunnersAlive;

		let foodLocations = getFoodLocations();

		for (let i = 23; i < 104; i++) {
			input_array[i] = foodLocations[i - 23];
		}

		let runnerInfo = getRunnerInfo();

		for (let i = 104; i < 124; i++) {
			input_array[i] = 0 //runnerInfo[i - 104];
		}

		this.action = this.brain.act(input_array);

		this.performAction(this.action);
	},
	backward: function() {
		var reward = (plDefX === 20) ? 1 : 0;
		//var reward = (baRunnersKilled === baTotalRunners) ? 1 : 0;
		this.last_reward = reward;
		this.brain.learn(reward);
	},
	performAction: function(a) {
		// if a % 13 === 0 : drop nothing
		if (a % 13 === 1) { // drop tofu
			dropTofuAction();
		} else if (a % 13 === 2) { // drop crackers
			dropCrackersAction();
		} else if (a % 13 === 3) { // drop worms
			dropWormsAction();
		} else if (a % 13 === 4) { // drop tofu, drop tofu
			dropTofuAction();
			dropTofuAction();
		} else if (a % 13 === 5) { // drop tofu, drop crackers
			dropTofuAction();
			dropCrackersAction();
		} else if (a % 13 === 6) { // drop tofu, drop worms
			dropTofuAction();
			dropWormsAction();
		} else if (a % 13 === 7) { // drop crackers, drop tofu
			dropCrackersAction();
			dropTofuAction();
		} else if (a % 13 === 8) { // drop crackers, drop crackers
			dropCrackersAction();
			dropCrackersAction();
		} else if (a % 13 === 9) { // drop crackers, drop worms
			dropCrackersAction();
			dropWormsAction();
		} else if (a % 13 === 10) { // drop worms, drop tofu
			dropWormsAction();
			dropTofuAction();
		} else if (a % 13 === 11) { // drop worms, drop crackers
			dropWormsAction();
			dropCrackersAction();
		} else if (a % 13 === 12) { // drop worms, drop worms
			dropWormsAction();
			dropWormsAction();
		}

		let primaryAction = Math.floor(a / 13);
		let dx = 0;
		let dy = 0;
		if (primaryAction === 0) { // pick up logs
			pickingUpLogs = true;
		} else if (primaryAction === 1) { // pick up hammer
			pickingUpHammer = true;
		} else if (primaryAction === 2) { // repair trap
			repairTrapAction();
		} else if (primaryAction === 3) { // pick up tofu
			pickingUpFood = "t";
		} else if (primaryAction === 4) { // pick up crackers
			pickingUpFood = "c";
		} else if (primaryAction === 5) { // pick up worms
			pickingUpFood = "w";
		} else if (primaryAction === 6) { // move center
			dx = 0;
			dy = 0;
		} else if (primaryAction === 7) { // move n
			dx = 0;
			dy = 1;
		} else if (primaryAction === 8) { // move nn
			dx = 0;
			dy = 2;
		} else if (primaryAction === 9) { // move ne
			dx = 1;
			dy = 1;
		} else if (primaryAction === 10) { // move nw
			dx = -1;
			dy = 1;
		} else if (primaryAction === 11) { // move nne
			dx = 1;
			dy = 2;
		} else if (primaryAction === 12) { // move nnw
			dx = -1;
			dy = 2;
		} else if (primaryAction === 13) { // move nee
			dx = 2;
			dy = 1;
		} else if (primaryAction === 14) { // move nww
			dx = -2;
			dy = 1;
		} else if (primaryAction === 15) { // move nnee
			dx = 2;
			dy = 2;
		} else if (primaryAction === 16) { // move nnww
			dx = -2;
			dy = 2;
		} else if (primaryAction === 17) { // move s
			dx = 0;
			dy = -1;
		} else if (primaryAction === 18) { // move ss
			dx = 0;
			dy = -2;
		} else if (primaryAction === 19) { // move se
			dx = 1;
			dy = -1;
		} else if (primaryAction === 20) { // move sw
			dx = -1;
			dy = -1;
		} else if (primaryAction === 21) { // move sse
			dx = 1;
			dy = -2;
		} else if (primaryAction === 22) { // move ssw
			dx = -1;
			dy = -2;
		} else if (primaryAction === 23) { // move see
			dx = 2;
			dy = -1;
		} else if (primaryAction === 24) { // move sww
			dx = -2;
			dy = -1;
		} else if (primaryAction === 25) { // move ssee
			dx = 2;
			dy = -2;
		} else if (primaryAction === 26) { // move ssww
			dx = -2;
			dy = -2;
		} else if (primaryAction === 27) { // move e
			dx = 1;
			dy = 0;
		} else if (primaryAction === 28) { // move ee
			dx = 2;
			dy = 0;
		} else if (primaryAction === 29) { // move w
			dx = -1;
			dy = 0;
		} else if (primaryAction === 30) { // move ww
			dx = -2;
			dy = 0;
		}

		plDefPathfind(plDefX + dx, plDefY + dy);
	}
}

var agent;
var env;
var spec;

// END OF SOME RL VARS

window.onload = simInit;
//{ Simulation - sim
function simInit() {
	let canvas = document.getElementById(HTML_CANVAS);
	simMovementsInput = document.getElementById(HTML_RUNNER_MOVEMENTS);
	simMovementsInput.onkeypress = function (e) {
		if (e.key === " ") {
			e.preventDefault();
		}
	};
	simTickDurationInput = document.getElementById(HTML_TICK_DURATION);
	simStartStopButton = document.getElementById(HTML_START_BUTTON);
	simStartStopButton.onclick = simStartStopButtonOnClick;
	simLoadAgentButton = document.getElementById(HTML_LOAD_AGENT);
	simLoadAgentButton.onclick = simLoadAgentButtonOnClick;
	simWaveSelect = document.getElementById(HTML_WAVE_SELECT);
	simWaveSelect.onchange = simWaveSelectOnChange;
	simDefLevelSelect = document.getElementById(HTML_DEF_LEVEL_SELECT);
	simToggleRepair = document.getElementById(HTML_TOGGLE_REPAIR);
	simToggleRepair.onchange = simToggleRepairOnChange;
	simTogglePauseSL = document.getElementById(HTML_TOGGLE_PAUSE_SL);
	simTogglePauseSL.onchange = simTogglePauseSLOnChange;
	simToggleInfiniteFood = document.getElementById(HTML_TOGGLE_INFINITE_FOOD);
	simToggleInfiniteFood.onchange = simToggleInfiniteFoodOnChange;
	simToggleLogHammerToRepair = document.getElementById(HTML_TOGGLE_LOG_HAMMER_TO_REPAIR);
	simToggleLogHammerToRepair.onchange = simToggleLogHammerToRepairOnChange;
	simDefLevelSelect.onchange = simDefLevelSelectOnChange;
	simTickCountSpan = document.getElementById(HTML_TICK_COUNT);
	currDefFoodSpan = document.getElementById(HTML_CURRENT_DEF_FOOD);
	rInit(canvas, 64*12, 48*12);
	rrInit(12);
	mInit(mWAVE_1_TO_9, 64, 48);
	ruInit(5);
	simReset();

	window.onkeydown = simWindowOnKeyDown;
	canvas.onmousedown = simCanvasOnMouseDown;
	canvas.oncontextmenu = function (e) {
		e.preventDefault();
	};

	//agent = new RunnerWave1Agent();
	agent = new Wave1TestAgent();

	env = agent;
	spec = {};
	spec.update = 'qlearn'; // qlearn | sarsa
	spec.gamma = 0.9; // discount factor, [0, 1)
	spec.epsilon = 0.2;//0.2 // initial epsilon for epsilon-greedy policy, [0, 1)
	spec.alpha = 0.005; // value function learning rate
	spec.experience_add_every = 5; // number of time steps before we add another experience to replay memory
	spec.experience_size = 10000; // size of experience
	spec.learning_steps_per_iteration = 5;
	spec.tderror_clamp = 1.0; // for robustness
	spec.num_hidden_units = 100; // number of neurons in hidden layer

	agent.brain = new RL.DQNAgent(env, spec);
}
function simReset() {
	if (simIsRunning) {
		clearInterval(simTickTimerId);
	}
	simIsRunning = false;
	simStartStopButton.innerHTML = "Start Wave";
	simLoadAgentButton.innerHTML = "Load Agent";
	baInit(0, 0, "");
	plDefInit(-1, 0);
	simDraw();
}
//const fs = require('fs');
function simLoadAgentButtonOnClick() {
	let data = {"nh":100,"ns":2,"na":3,"net":{"W1":{"n":100,"d":2,"w":{"0":1.7686691385077302,"1":-1.3509743826589413,"2":0.8972628411656992,"3":-0.6093894394701163,"4":-0.8325602874527979,"5":0.585832651689498,"6":-0.5111710490978799,"7":0.2705440163770432,"8":0.3392425244918458,"9":-0.18299261365387828,"10":-0.9247288118352122,"11":0.6667302831270634,"12":-1.1311068204247023,"13":0.8480268639121588,"14":0.9685692675064557,"15":-0.7113545938837821,"16":0.6577310416826124,"17":-0.44963138786714524,"18":2.8926674037231432,"19":-3.560351174848016,"20":0.7175313968757325,"21":-0.49163006628777417,"22":-1.6079253725474614,"23":1.2621907999340276,"24":0.9448919922413718,"25":-0.6911467873431919,"26":-0.7680260384296174,"27":0.47394229184612663,"28":-0.2698880532723172,"29":0.09374398079700605,"30":-0.6813777654958622,"31":0.4768506642775065,"32":1.2826022524405671,"33":-0.9768296822138121,"34":-0.9267451488597443,"35":0.6722714972888205,"36":1.3125379516732443,"37":-1.009127251370418,"38":0.646597334322368,"39":-0.4417090973083799,"40":-0.8311577744557481,"41":0.5891038743436822,"42":0.8971126495755443,"43":-0.6438268483564502,"44":0.7506930698054692,"45":-0.5343139478119953,"46":0.8491899551326071,"47":-0.6081255113122203,"48":-0.18478357251409402,"49":-0.18898215815456224,"50":0.4725688672142496,"51":-0.27218278382221167,"52":0.30891798184050656,"53":-0.12704868262018032,"54":-0.6011206445309976,"55":0.3427438523548556,"56":-0.41590972204890664,"57":0.17734764670316674,"58":-0.9978573519144442,"59":0.7313626639998111,"60":-0.7471291604016052,"61":0.4637158254805591,"62":-0.9884533819285775,"63":0.7285909640792585,"64":0.7855957764330965,"65":-0.5150871273920706,"66":-0.02961518748213103,"67":-0.6698046294747471,"68":-0.4618801661050382,"69":0.23635438506835513,"70":-0.5338307700282542,"71":0.2819400496540168,"72":-2.521925686948492,"73":2.0021991363352893,"74":-2.099178838462869,"75":1.699204321561906,"76":0.8757955426346304,"77":-0.6284107656050512,"78":0.3711919667134208,"79":-0.12608326885739257,"80":1.0242949805269215,"81":-0.7031253618347018,"82":-0.28798399631841726,"83":0.12565276488746904,"84":0.8224276669081768,"85":-0.5821446233598245,"86":0.730058251191718,"87":-0.5114591891416531,"88":0.7995977114464068,"89":-0.564368704117745,"90":0.8130014712960905,"91":-0.5953907619858537,"92":-0.9735212015615093,"93":0.6626206868429396,"94":-1.0366888207284495,"95":0.7356060087607821,"96":-1.3488791981288935,"97":1.041957444347494,"98":-0.8122273505703886,"99":0.5406011748295432,"100":0.7068983392621819,"101":-0.4416779145460985,"102":-0.4003024712994705,"103":0.19219084917973103,"104":-0.37542154191579585,"105":0.20760945812309106,"106":1.1644716884405055,"107":-0.8753905002950091,"108":0.7972038234201461,"109":-0.5231912380714118,"110":0.6216739612964599,"111":-0.40366018837127426,"112":-0.9643779814896363,"113":0.7019787163494521,"114":-0.9234705254686126,"115":0.6125083838132769,"116":0.2680623005746342,"117":-0.08288738863143978,"118":1.0926587078324654,"119":-0.7595934152460813,"120":0.08734581979640972,"121":-1.1163108528601147,"122":-0.8967838142029876,"123":0.668021391844382,"124":0.8799497352565516,"125":-0.6444464060627751,"126":0.934437417803329,"127":-0.6598461805137728,"128":-0.9885807082014163,"129":0.7226966571146566,"130":0.8043461028889947,"131":-0.52246085285188,"132":-0.8851807577633817,"133":0.6342871705516914,"134":-0.48545223579024277,"135":0.24593056660137116,"136":0.6632216764273692,"137":-0.44949652057086653,"138":0.847111477296057,"139":-0.6071544312198895,"140":-0.3824417425867659,"141":0.18835280852943137,"142":1.4821327794481103,"143":-1.097778459489924,"144":0.842879404714131,"145":-0.5437817845243113,"146":0.3445885409290197,"147":-0.12133025273088854,"148":1.0048953961573999,"149":-0.7389180800533567,"150":-0.8889052091433776,"151":0.6431065661011466,"152":-1.1891137907609863,"153":0.8936487258928905,"154":-2.143923263663479,"155":1.7034064530727302,"156":-1.5277002528838854,"157":1.2116897027790468,"158":-1.3985239697445524,"159":1.039339354672856,"160":0.7694683602036536,"161":-0.543062715549996,"162":-0.3207317900204909,"163":0.12731500935852338,"164":-0.6121209219397552,"165":0.46594868711049287,"166":0.7658149166907521,"167":-0.4832495352782254,"168":1.16748530622604,"169":-0.8816730596030986,"170":-0.8226605207540302,"171":0.582723104028251,"172":2.180551963807934,"173":-1.709774850919218,"174":-0.9470930171215144,"175":0.6557633124248252,"176":0.8567193784127423,"177":-0.6133532615648694,"178":-0.7264195465485533,"179":0.5267555138990674,"180":-1.022969801474229,"181":0.7562216945099024,"182":0.843539872658659,"183":-0.6029568734685691,"184":0.3762242195306292,"185":-0.13537058775686014,"186":1.2330438466873501,"187":-0.9362329090460044,"188":-0.415446757715514,"189":0.18992948766528395,"190":0.8928636660748148,"191":-0.6567046853644329,"192":0.7559863454235641,"193":-0.5264079021130015,"194":0.2697733810134778,"195":-0.12762899980785006,"196":-0.7972529575511078,"197":0.5664307046570213,"198":1.9885032705393102,"199":-1.5789333014071312}},"b1":{"n":100,"d":1,"w":{"0":0.05382154621130466,"1":0.02697693615659217,"2":-0.024772239883103404,"3":-0.015539803412467138,"4":0.01090043689901945,"5":-0.02750573088565909,"6":-0.03424409831012693,"7":0.029046455148248523,"8":0.01925306991237558,"9":0.08732849891543451,"10":0.021620914605659226,"11":-0.04888419778242052,"12":0.028795618400991138,"13":-0.022939343210355477,"14":-0.008058918735565882,"15":-0.02068203460939037,"16":0.03886821923971765,"17":-0.02870698798875718,"18":0.03910506158935402,"19":0.019574050696554702,"20":-0.02538137963097739,"21":0.027156752816169426,"22":0.02270353713080726,"23":0.025823923903971306,"24":-0.005377407433613239,"25":0.014309537917519623,"26":0.0092965341106165,"27":-0.017987552703161987,"28":-0.012806190958626909,"29":-0.03004116469599781,"30":-0.022131904670887364,"31":-0.030263889120693083,"32":0.024291451554796315,"33":-0.0005274678602648842,"34":-0.0141423210287536,"35":-0.016213535623313898,"36":-0.07715529267315753,"37":-0.06391377433337561,"38":0.026594638806486198,"39":0.011595334291810576,"40":0.03149377689480911,"41":-0.008572195474988404,"42":0.025425631714001824,"43":0.02215926176579269,"44":0.02476226610516513,"45":0.024671786329737884,"46":-0.029491996859640265,"47":-0.03148294422267529,"48":-0.04106281799561849,"49":-0.023576793136382865,"50":0.02118936572584651,"51":-0.01195205106452103,"52":-0.012101948725996835,"53":0.035526602151032025,"54":0.024545808430048548,"55":0.01871369018179502,"56":-0.029064712133443634,"57":-0.027888903985354697,"58":0.008196956467945915,"59":0.032907707085175875,"60":0.002579231694952808,"61":-0.027060463260686642,"62":0.026557496445313074,"63":0.028351110881483583,"64":-0.029629386820183345,"65":0.024303481611028868,"66":-0.026471433302675308,"67":-0.01554113077780965,"68":0.02033822673196866,"69":0.025450056111450407,"70":-0.011972416914327922,"71":0.04505357709572033,"72":0.025317442894440215,"73":0.010739828881229188,"74":0.0303784086977341,"75":-0.02707728060206731,"76":-0.03584341158071692,"77":-0.06520988131443374,"78":-0.04656444588795441,"79":-0.042467155068592555,"80":0.023166587178074734,"81":-0.010328636802244192,"82":-0.018526440621756218,"83":0.023353007095933416,"84":0.03532575728832253,"85":-0.024368620795798303,"86":0.06648195486072196,"87":-0.027971546534261715,"88":0.025757977378490622,"89":-0.022588941637681038,"90":-0.03145752420310643,"91":0.025541605828333553,"92":0.011513839415291723,"93":0.036970401145845315,"94":-0.012605990856484728,"95":0.02741584319128947,"96":0.022882307998870345,"97":0.00842896885010921,"98":-0.02401498773499679,"99":0.06021900243397062}},"W2":{"n":3,"d":100,"w":{"0":0.09487928979887164,"1":0.3355089143482673,"2":-0.0812638213970898,"3":-0.5256635361089573,"4":-0.5556314085481892,"5":0.40394423036231436,"6":0.5571349948395765,"7":-0.17780244709483806,"8":-0.11364117036506631,"9":0.11586744483910427,"10":-0.2872104308251436,"11":0.6100759770287104,"12":-0.18305592638807922,"13":-0.2759535010463474,"14":0.09472110007453965,"15":-0.016731162977126464,"16":-0.5293181199334227,"17":0.4116757975694604,"18":-0.6048055570231042,"19":-0.09422899828472589,"20":0.41075977840586414,"21":-0.41038643850247974,"22":-0.018541210133521266,"23":-0.12807433475384888,"24":-1.562718641541767,"25":-0.13920706185792092,"26":-0.2023750976093037,"27":-0.4536088882542191,"28":-0.45024887701791905,"29":0.3637608347317815,"30":-0.43643606880446556,"31":0.46407289994910866,"32":0.1707869333012293,"33":0.2089028947321831,"34":-0.5579447786043734,"35":-0.45691492569619097,"36":-1.500109854939811,"37":-0.41644944027028813,"38":-0.40959360448330273,"39":0.41203146940504604,"40":0.23632590530434466,"41":0.2964731717704809,"42":-0.32566849764443867,"43":-0.13688030444809343,"44":-0.3163635176541344,"45":0.2742399129037771,"46":-0.44155541364721257,"47":-0.420659724513906,"48":0.5299835329881291,"49":-0.6227616844401728,"50":0.30447270938819504,"51":0.17314421282247427,"52":0.8260879073093176,"53":-0.5487560314636256,"54":0.1705588495709553,"55":-0.2289063365255088,"56":0.325010402561942,"57":-0.2597777685742738,"58":0.013182479940551052,"59":0.3756548825156285,"60":0.027923660891345483,"61":0.0219577271502901,"62":-0.1294359244021419,"63":0.015357868215760925,"64":0.4192313947562617,"65":0.5402171716317364,"66":0.10463693893240507,"67":-0.5250060662546585,"68":-0.26285870391994887,"69":-0.22928784920873363,"70":0.5478599338455971,"71":0.3213973562279058,"72":0.3927811101599762,"73":-0.024133146983445302,"74":-0.29084892934647777,"75":0.10754223100089437,"76":-0.2088705773922546,"77":0.7413799171059545,"78":-0.7483407322795476,"79":-0.27892432541790196,"80":-0.1658764640906888,"81":0.08356356760987216,"82":-0.23612338286768533,"83":0.4147308390929283,"84":-0.5252555307222813,"85":0.25859555363820713,"86":1.1018205600782733,"87":-0.20787053283697315,"88":-0.31662111613995997,"89":-0.07331397807872121,"90":0.4226626019623909,"91":-0.2545088563647179,"92":0.0633054539253523,"93":-0.525802665088035,"94":-0.561124173989919,"95":-0.035133144182778596,"96":-0.11889224391450286,"97":-0.2527609283695178,"98":0.3523870595819859,"99":0.1936286007517183,"100":0.2113424570951516,"101":0.12151754797601108,"102":-0.016497920523794705,"103":-0.4110457067539475,"104":-0.5103715461700739,"105":0.3624884220949662,"106":0.46706843776661544,"107":-0.17067052236728616,"108":-0.09296783990175785,"109":0.11813182595413084,"110":-0.2528791143298341,"111":0.5055996271289357,"112":-0.17254828450556445,"113":-0.26733494097314653,"114":0.06104451767045143,"115":-0.03588620039496104,"116":-0.434743185728584,"117":0.2904652666531264,"118":-0.5158491069702467,"119":-0.07504435825311795,"120":0.3564984183663863,"121":-0.33038168641958227,"122":-0.00689667569214104,"123":-0.0820035341499265,"124":-1.3296039317965633,"125":-0.13481435835816627,"126":-0.13547772619979354,"127":-0.38587892305189725,"128":-0.36371799890300244,"129":0.32332696349965867,"130":-0.28632110607182226,"131":0.38326617380233935,"132":0.13612803439221474,"133":0.12481841171524961,"134":-0.42770209389168956,"135":-0.4173416938844734,"136":-1.0792364811467312,"137":-0.3736948683245353,"138":-0.32684697448949074,"139":0.24528011628246965,"140":0.21392783033481005,"141":0.21660263156831397,"142":-0.28484800725522924,"143":-0.13164488167062588,"144":-0.28653588679557285,"145":0.15495529860388124,"146":-0.45206499540059086,"147":-0.39568788957577605,"148":0.44649828350250226,"149":-0.5464795893657014,"150":0.15888159158523965,"151":0.16853330657211515,"152":0.734814447138278,"153":-0.40684850113546317,"154":0.13139174683640406,"155":-0.23156893758647282,"156":0.267501530588541,"157":-0.17811426393888644,"158":0.008871129596563943,"159":0.33086261265103334,"160":-0.008502912812532525,"161":-0.025778957323210186,"162":-0.14339245075106383,"163":0.13500148402656578,"164":0.39945455221862475,"165":0.40276270219519666,"166":0.05667832900844599,"167":-0.37730010446354517,"168":-0.2400291879953444,"169":-0.20384611046061915,"170":0.4897726686153305,"171":0.21880885254277788,"172":0.4139908991954671,"173":-0.015781660575548875,"174":-0.2326531889310527,"175":0.042353866096296586,"176":-0.29836683583077317,"177":0.5453633928517828,"178":-0.5835282424328839,"179":-0.39609411057717253,"180":-0.15513915211760967,"181":0.054137324617028956,"182":-0.21132147246829808,"183":0.4249465091740853,"184":-0.46950972955289605,"185":0.22551116471933438,"186":0.8757229482397476,"187":-0.21858676585292877,"188":-0.2628112752288246,"189":-0.10306526554922751,"190":0.3646480276144266,"191":-0.20886343475091187,"192":0.05916737147877671,"193":-0.4536522882720143,"194":-0.4672445015801018,"195":-0.01821448727676269,"196":-0.10128945757379788,"197":-0.1975283773795532,"198":0.3402114006478605,"199":0.26854001707644487,"200":0.15506199014296457,"201":0.27904985109976926,"202":-0.101833039901569,"203":-0.4799852700769914,"204":-0.5617743378294859,"205":0.42195539153631473,"206":0.42882743020020464,"207":-0.15350673273425583,"208":-0.09112782970785274,"209":0.11813063190918496,"210":-0.2943902576534996,"211":0.4335670744719278,"212":-0.16410889669517245,"213":-0.2406463040880852,"214":0.13917821312837417,"215":0.005883148263670569,"216":-0.467621792311882,"217":0.3257778069073786,"218":-0.43767693043634004,"219":-0.06766312634222467,"220":0.363640738145487,"221":-0.4075167515672287,"222":-0.02064668618648403,"223":-0.097760752398526,"224":-1.5480503171183224,"225":-0.15151737834107395,"226":-0.2565089867866414,"227":-0.3578972986690518,"228":-0.42563049363223976,"229":0.32739944948478844,"230":-0.4251875197563234,"231":0.3035363067126687,"232":0.2385143751553803,"233":0.22962305034936373,"234":-0.49290868433708096,"235":-0.3787687320084377,"236":-1.24807344616026,"237":-0.4199048764012137,"238":-0.34175075253549303,"239":0.35034396957061814,"240":0.17455773393744628,"241":0.32619003676687336,"242":-0.28208690185192803,"243":-0.08716829711076432,"244":-0.281165896166046,"245":0.28683889837273585,"246":-0.30110768750982825,"247":-0.336831252218911,"248":0.36595396998849494,"249":-0.4644138101529646,"250":0.28052653282647716,"251":0.16525665031663492,"252":0.8352795905979845,"253":-0.4497824262657257,"254":0.15361767137645266,"255":-0.21898574909681393,"256":0.290723265761372,"257":-0.19810628062477306,"258":-0.021403880592453402,"259":0.30010405001729523,"260":0.10672156491044307,"261":-0.04679660395128467,"262":-0.10937501581697602,"263":-0.04142536331936369,"264":0.39082567503095095,"265":0.4887010282311839,"266":0.1280213432242454,"267":-0.48094741553510056,"268":-0.20462792014811987,"269":-0.1909167011960334,"270":0.5106311791586162,"271":0.34766677730907053,"272":0.26951540642103294,"273":-0.07237752870240656,"274":-0.2715990794817476,"275":0.09004326682167543,"276":-0.15628101755152324,"277":0.5549766191154938,"278":-0.6777299731751526,"279":-0.20620459382313633,"280":-0.14812148031217598,"281":0.086708548534637,"282":-0.2769222353346122,"283":0.28940881701567517,"284":-0.43550527989428145,"285":0.25733647391775266,"286":0.9665710324990289,"287":-0.2559763941332242,"288":-0.2717598069156352,"289":-0.05083040394794037,"290":0.36528514483144164,"291":-0.2199384059057956,"292":0.010234172347682268,"293":-0.47060141939647093,"294":-0.4839321931783828,"295":-0.026271561640313894,"296":-0.1390303503633807,"297":-0.2604500060907967,"298":0.2876625539443432,"299":0.272547394579884}},"b2":{"n":3,"d":1,"w":{"0":1.6945134370848944,"1":1.424268576044991,"2":1.619562605391085}}}};
	agent.brain.fromJSON(data); // corss your fingers...
	// set epsilon to be much lower for more optimal behavior
	agent.brain.epsilon = 0.0;
	// kill learning rate to not learn
	agent.brain.alpha = 0.0;
	console.log("AGENT LOADED");

	//agent.brain.fromJSON(jsonData); // node.js
}
function simStartStopButtonOnClick() {
	if (simIsRunning) {
		mResetMap();
		simReset();
	} else {
		let movements = simParseMovementsInput();
		if (movements === null) {
			alert("Invalid runner movements. Example: ws-s");
			return;
		}
		simIsRunning = true;
		simStartStopButton.innerHTML = "Stop Wave";
		let maxRunnersAlive = 0;
		let totalRunners = 0;
		let wave = simWaveSelect.value;
		switch(Number(wave)) {
		case 1:
			maxRunnersAlive = 2;
			totalRunners = 2;
			break;
		case 2:
			maxRunnersAlive = 2;
			totalRunners = 3;
			break;
		case 3:
			maxRunnersAlive = 2;
			totalRunners = 4;
			break;
		case 4:
			maxRunnersAlive = 3;
			totalRunners = 4;
			break;
		case 5:
			maxRunnersAlive = 4;
			totalRunners = 5;
			break;
		case 6:
			maxRunnersAlive = 4;
			totalRunners = 6;
			break;
		case 7:
		case 10:
			maxRunnersAlive = 5;
			totalRunners = 6;
			break;
		case 8:
			maxRunnersAlive = 5;
			totalRunners = 7;
			break;
		case 9:
			maxRunnersAlive = 5;
			totalRunners = 9;
			break;
		}
		baInit(maxRunnersAlive, totalRunners, movements);
		if (mCurrentMap === mWAVE10) {
			plDefInit(baWAVE10_DEFENDER_SPAWN_X, baWAVE10_DEFENDER_SPAWN_Y);
		} else {
			plDefInit(baWAVE1_DEFENDER_SPAWN_X, baWAVE1_DEFENDER_SPAWN_Y);
		}
		console.log("Wave " + wave + " started!");
		rlTick();
		simTickTimerId = setInterval(rlTick, Number(simTickDurationInput.value));
		//simTick();
		//simTickTimerId = setInterval(simTick, Number(simTickDurationInput.value)); // tick time in milliseconds (set to 600 for real)
	}
}
function simParseMovementsInput() {
	let movements = simMovementsInput.value.split("-");
	for (let i = 0; i < movements.length; ++i) {
		let moves = movements[i];
		for (let j = 0; j < moves.length; ++j) {
			let move = moves[j];
			if (move !== "" && move !== "s" && move !== "w" && move !== "e") {
				return null;
			}
		}
	}
	return movements;
}
function simWindowOnKeyDown(e) {
	if (simIsRunning) {
		if (e.key === "t" && (numTofu > 0 || infiniteFood === "yes") && repairTicksRemaining === 0) {
			numTofu -= 1;
			mAddItem(new fFood(plDefX, plDefY, currDefFood === "t", "t"));
		} else if (e.key === "c" && (numCrackers > 0 || infiniteFood === "yes") && repairTicksRemaining === 0) {
			numCrackers -= 1;
			mAddItem(new fFood(plDefX, plDefY, currDefFood === "c", "c"));
		} else if (e.key === "w" && (numWorms > 0 || infiniteFood === "yes") && repairTicksRemaining === 0) {
			numWorms -= 1;
			mAddItem(new fFood(plDefX, plDefY, currDefFood === "w", "w"));
		} else if (e.key === "1") {
			pickingUpFood = "t";
		} else if (e.key === "2") {
			pickingUpFood = "c";
		} else if (e.key === "3") {
			pickingUpFood = "w";
		} else if (e.key === "l") {
			pickingUpLogs = true;
		} else if (e.key === "h") {
			pickingUpHammer = true;
		} else if (e.key === "r") {
			if (repairTicksRemaining === 0 && ((isInEastRepairRange(plDefX, plDefY) && eastTrapState < 2 ) || (isInWestRepairRange(plDefX, plDefY) && westTrapState < 2))) {
				if ((hasHammer && numLogs > 0) || logHammerToRepair === "no") {
					repairTicksRemaining = 5;
					if (plDefStandStillCounter === 0) {
						++repairTicksRemaining;
					}
				}
			}
		} else if (e.key === "p") {
			isPaused = !isPaused;
		} else if (e.key === "s") {
			if (isPaused || pauseSL !== "yes") {
				isPaused = true;
				saveGameState();
				saveExists = true;
			}
		} else if (e.key === "y" && saveExists) {
			if (isPaused || pauseSL !== "yes") {
				isPaused = true;
				loadGameState();
			}
		}
	}
	if (e.key === " ") {
		simStartStopButtonOnClick();
		e.preventDefault();
	}
}

function repairTrapAction() {
	if (repairTicksRemaining === 0 && ((isInEastRepairRange(plDefX, plDefY) && eastTrapState < 2 ) || (isInWestRepairRange(plDefX, plDefY) && westTrapState < 2))) {
		if ((hasHammer && numLogs > 0) || logHammerToRepair === "no") {
			repairTicksRemaining = 5;
			if (plDefStandStillCounter === 0) {
				++repairTicksRemaining;
			}
		}
	}
}

function dropTofuAction() {
	if ((numTofu > 0 || infiniteFood === "yes") && repairTicksRemaining === 0) {
		numTofu -= 1;
		mAddItem(new fFood(plDefX, plDefY, currDefFood === "t", "t"));
	}
}

function dropCrackersAction() {
	if ((numCrackers > 0 || infiniteFood === "yes") && repairTicksRemaining === 0) {
		numCrackers -= 1;
		mAddItem(new fFood(plDefX, plDefY, currDefFood === "c", "c"));
	}
}

function dropWormsAction() {
	if ((numWorms > 0 || infiniteFood === "yes") && repairTicksRemaining === 0) {
		numWorms -= 1;
		mAddItem(new fFood(plDefX, plDefY, currDefFood === "w", "w"));
	}
}

var saveExists = false;
var isPaused; // true/false

function simCanvasOnMouseDown(e) {
	var canvasRect = rCanvas.getBoundingClientRect();
	let xTile = Math.trunc((e.clientX - canvasRect.left) / rrTileSize);
	let yTile = Math.trunc((canvasRect.bottom - 1 - e.clientY) / rrTileSize);
	if (e.button === 0) {
		plDefPathfind(xTile, yTile);
	} else if (e.button === 2) {
		if (xTile === baCollectorX && yTile === baCollectorY) {
			baCollectorX = -1;
		} else {
			baCollectorX = xTile;
			baCollectorY = yTile;
		}
	}
}
//*/

function simWaveSelectOnChange(e) {
	if (simWaveSelect.value === "10") {
		mInit(mWAVE10, 64, 48);
	} else {
		mInit(mWAVE_1_TO_9, 64, 48);
	}
	simReset();
}
function simDefLevelSelectOnChange(e) {
	mResetMap();
	simReset();
	ruInit(Number(simDefLevelSelect.value));
}
function simToggleRepairOnChange(e) {
	requireRepairs = simToggleRepair.value;
}
function simTogglePauseSLOnChange(e) {
	pauseSL = simTogglePauseSL.value;
}
function simToggleInfiniteFoodOnChange(e) {
	infiniteFood = simToggleInfiniteFood.value;
}
function simToggleLogHammerToRepairOnChange(e) {
	logHammerToRepair = simToggleLogHammerToRepair.value;
}
//*/
function simTick() {
	if (!isPaused) {
		baTick();
		plDefTick();
		simDraw();
	}
}
function simDraw() {
	mDrawMap();
	baDrawDetails();
	mDrawItems();
	baDrawEntities();
	plDefDrawPlayer();
	mDrawGrid();
	baDrawOverlays();
	rPresent();
}
var simTickTimerId;
var simMovementsInput;
var simStartStopButton;
var simLoadAgentButton;
var simWaveSelect;
var simDefLevelSelect;
var simToggleRepair;
var simTickCountSpan;
var simIsRunning;
var currDefFoodSpan;
var simTickDurationInput;
var simTogglePauseSL;
var simToggleInfiniteFood;
var simToggleLogHammerToRepair;

var numTofu; // 0-9
var numCrackers; // 0-9
var numWorms; // 0-9
var currDefFood; // "t", "c", "w"
var numLogs; // 0-27
var hasHammer; // true/false
var eastTrapState; // less than 2 (can be negative)
var westTrapState; // less than 2 (can be negative)
var northwestLogsState; // true/false
var southeastLogsState; // true/false
var hammerState; // true/false

var requireRepairs;
var pauseSL;
var infiniteFood;
var logHammerToRepair;

var pickingUpFood; // "t", "c", "w", "n"
var pickingUpLogs; // true/false
var pickingUpHammer; // true/false
var repairTicksRemaining; // 0-5
//}
//{ PlayerDefender - plDef
function plDefInit(x, y) {
	plDefX = x;
	plDefY = y;
	pickingUpFood = "n";
	pickingUpLogs = false;
	pickingUpHammer = false;
	repairTicksRemaining = 0;
	plDefPathQueuePos = 0;
	plDefPathQueueX = [];
	plDefPathQueueY = [];
	plDefShortestDistances = [];
	plDefWayPoints = [];
	plDefStandStillCounter = 0;
}
function plDefTick() {
	++plDefStandStillCounter;
	let prevX = plDefX;
	let prevY = plDefY;
	if (repairTicksRemaining > 0) {
		if (repairTicksRemaining === 1) {
			numLogs -=1;
			if (isInEastRepairRange(plDefX, plDefY)) {
				eastTrapState = 2;
			} else {
				westTrapState = 2;
			}
		}
		repairTicksRemaining -= 1;
		plDefPathQueuePos = 0;
		pickingUpFood = "n";
	} else if (pickingUpFood !== "n") {
		let itemZone = mGetItemZone(plDefX >>> 3, plDefY >>> 3);
		for (let i = 0; i < itemZone.length; ++i) {
			let item = itemZone[i];
			if (plDefX === item.x && plDefY === item.y && item.type === pickingUpFood) {
				itemZone.splice(i, 1);
				if (pickingUpFood === "t") {
					numTofu += 1;
				} else if (pickingUpFood === "c") {
					numCrackers += 1;
				} else {
					numWorms += 1;
				}
				break;
			}
		}
		pickingUpFood = "n";
		plDefPathQueuePos = 0;
	} else if (pickingUpLogs) {
		let waveIs10 = mCurrentMap === mWAVE10;
		if ((waveIs10 && plDefX === WAVE10_NW_LOGS_X && plDefY === WAVE10_NW_LOGS_Y) || (!waveIs10 && plDefX === WAVE1_NW_LOGS_X && plDefY === WAVE1_NW_LOGS_Y)) {
			if (northwestLogsState) {
				numLogs += 1;
				northwestLogsState = false;
			}
		}  else if ((waveIs10 && plDefX === WAVE10_SE_LOGS_X && plDefY === WAVE10_SE_LOGS_Y) || (!waveIs10 && plDefX === WAVE1_SE_LOGS_X && plDefY === WAVE1_SE_LOGS_Y)) {
			if (southeastLogsState) {
				numLogs += 1;
				southeastLogsState = false;
			}
		}
		pickingUpLogs = false;
	} else if (pickingUpHammer) {
		if (hammerState && plDefX === HAMMER_X && plDefY === HAMMER_Y) {
			hasHammer = true;
			hammerState = false;
		}
		pickingUpHammer = false;
	} else if (plDefPathQueuePos > 0) {
		plDefX = plDefPathQueueX[--plDefPathQueuePos];
		plDefY = plDefPathQueueY[plDefPathQueuePos];
		if (plDefPathQueuePos > 0) {
			plDefX = plDefPathQueueX[--plDefPathQueuePos];
			plDefY = plDefPathQueueY[plDefPathQueuePos];
		}
	}
	if (prevX !== plDefX || prevY !== plDefY) {
		plDefStandStillCounter = 0;
	}
}
function plDefDrawPlayer() {
	if (plDefX >= 0) {
		rSetDrawColor(240, 240, 240, 200);
		rrFill(plDefX, plDefY);
	}
}
function plDefPathfind(destX, destY) {
	for (let i = 0; i < mWidthTiles*mHeightTiles; ++i) {
		plDefShortestDistances[i] = 99999999;
		plDefWayPoints[i] = 0;
	}
	plDefWayPoints[plDefX + plDefY*mWidthTiles] = 99;
	plDefShortestDistances[plDefX + plDefY*mWidthTiles] = 0;
	plDefPathQueuePos = 0;
	let pathQueueEnd = 0;
	plDefPathQueueX[pathQueueEnd] = plDefX;
	plDefPathQueueY[pathQueueEnd++] = plDefY;
	let currentX;
	let currentY;
	let foundDestination = false;
	while (plDefPathQueuePos !== pathQueueEnd) {
		currentX = plDefPathQueueX[plDefPathQueuePos];
		currentY = plDefPathQueueY[plDefPathQueuePos++];
		if (currentX === destX && currentY === destY) {
			foundDestination = true;
			break;
		}
		let newDistance = plDefShortestDistances[currentX + currentY*mWidthTiles] + 1;
		let index = currentX - 1 + currentY*mWidthTiles;
		if (currentX > 0 && plDefWayPoints[index] === 0 && (mCurrentMap[index] & 19136776) === 0) {
			plDefPathQueueX[pathQueueEnd] = currentX - 1;
			plDefPathQueueY[pathQueueEnd++] = currentY;
			plDefWayPoints[index] = 2;
			plDefShortestDistances[index] = newDistance;
		}
		index = currentX + 1 + currentY*mWidthTiles;
		if (currentX < mWidthTiles - 1 && plDefWayPoints[index] === 0 && (mCurrentMap[index] & 19136896) === 0) {
			plDefPathQueueX[pathQueueEnd] = currentX + 1;
			plDefPathQueueY[pathQueueEnd++] = currentY;
			plDefWayPoints[index] = 8;
			plDefShortestDistances[index] = newDistance;
		}
		index = currentX + (currentY - 1)*mWidthTiles;
		if (currentY > 0 && plDefWayPoints[index] === 0 && (mCurrentMap[index] & 19136770) === 0) {
			plDefPathQueueX[pathQueueEnd] = currentX;
			plDefPathQueueY[pathQueueEnd++] = currentY - 1;
			plDefWayPoints[index] = 1;
			plDefShortestDistances[index] = newDistance;
		}
		index = currentX + (currentY + 1)*mWidthTiles;
		if (currentY < mHeightTiles - 1 && plDefWayPoints[index] === 0 && (mCurrentMap[index] & 19136800) === 0) {
			plDefPathQueueX[pathQueueEnd] = currentX;
			plDefPathQueueY[pathQueueEnd++] = currentY + 1;
			plDefWayPoints[index] = 4;
			plDefShortestDistances[index] = newDistance;
		}
		index = currentX - 1 + (currentY - 1)*mWidthTiles;
		if (currentX > 0 && currentY > 0 && plDefWayPoints[index] === 0 &&
		(mCurrentMap[index] & 19136782) == 0 &&
		(mCurrentMap[currentX - 1 + currentY*mWidthTiles] & 19136776) === 0 &&
		(mCurrentMap[currentX + (currentY - 1)*mWidthTiles] & 19136770) === 0) {
			plDefPathQueueX[pathQueueEnd] = currentX - 1;
			plDefPathQueueY[pathQueueEnd++] = currentY - 1;
			plDefWayPoints[index] = 3;
			plDefShortestDistances[index] = newDistance;
		}
		index = currentX + 1 + (currentY - 1)*mWidthTiles;
		if (currentX < mWidthTiles - 1 && currentY > 0 && plDefWayPoints[index] === 0 &&
		(mCurrentMap[index] & 19136899) == 0 &&
		(mCurrentMap[currentX + 1 + currentY*mWidthTiles] & 19136896) === 0 &&
		(mCurrentMap[currentX + (currentY - 1)*mWidthTiles] & 19136770) === 0) {
			plDefPathQueueX[pathQueueEnd] = currentX + 1;
			plDefPathQueueY[pathQueueEnd++] = currentY - 1;
			plDefWayPoints[index] = 9;
			plDefShortestDistances[index] = newDistance;
		}
		index = currentX - 1 + (currentY + 1)*mWidthTiles;
		if (currentX > 0 && currentY < mHeightTiles - 1 && plDefWayPoints[index] === 0 &&
		(mCurrentMap[index] & 19136824) == 0 &&
		(mCurrentMap[currentX - 1 + currentY*mWidthTiles] & 19136776) === 0 &&
		(mCurrentMap[currentX + (currentY + 1)*mWidthTiles] & 19136800) === 0) {
			plDefPathQueueX[pathQueueEnd] = currentX - 1;
			plDefPathQueueY[pathQueueEnd++] = currentY + 1;
			plDefWayPoints[index] = 6;
			plDefShortestDistances[index] = newDistance;
		}
		index = currentX + 1 + (currentY + 1)*mWidthTiles;
		if (currentX < mWidthTiles - 1 && currentY < mHeightTiles - 1 && plDefWayPoints[index] === 0 &&
		(mCurrentMap[index] & 19136992) == 0 &&
		(mCurrentMap[currentX + 1 + currentY*mWidthTiles] & 19136896) === 0 &&
		(mCurrentMap[currentX + (currentY + 1)*mWidthTiles] & 19136800) === 0) {
			plDefPathQueueX[pathQueueEnd] = currentX + 1;
			plDefPathQueueY[pathQueueEnd++] = currentY + 1;
			plDefWayPoints[index] = 12;
			plDefShortestDistances[index] = newDistance;
		}
	}
	if (!foundDestination) {
		let bestDistanceStart = 0x7FFFFFFF;
		let bestDistanceEnd = 0x7FFFFFFF;
		let deviation = 10;
		for (let x = destX - deviation; x <= destX + deviation; ++x) {
			for (let y = destY - deviation; y <= destY + deviation; ++y) {
				if (x >= 0 && y >= 0 && x < mWidthTiles && y < mHeightTiles) {
					let distanceStart = plDefShortestDistances[x + y*mWidthTiles];
					if (distanceStart < 100) {
						let dx = Math.max(destX - x);
						let dy = Math.max(destY - y);
						let distanceEnd = dx*dx + dy*dy;
						if (distanceEnd < bestDistanceEnd || (distanceEnd === bestDistanceEnd && distanceStart < bestDistanceStart)) {
							bestDistanceStart = distanceStart;
							bestDistanceEnd = distanceEnd;
							currentX = x;
							currentY = y;
							foundDestination = true;
						}
					}
				}
			}
		}
		if (!foundDestination) {
			plDefPathQueuePos = 0;
			return;
		}
	}
	plDefPathQueuePos = 0;
	while (currentX !== plDefX || currentY !== plDefY) {
		let waypoint = plDefWayPoints[currentX + currentY*mWidthTiles];
		plDefPathQueueX[plDefPathQueuePos] = currentX;
		plDefPathQueueY[plDefPathQueuePos++] = currentY;
		if ((waypoint & 2) !== 0) {
			++currentX;
		} else if ((waypoint & 8) !== 0) {
			--currentX;
		}
		if ((waypoint & 1) !== 0) {
			++currentY;
		} else if ((waypoint & 4) !== 0) {
			--currentY;
		}
	}
}
var plDefPathQueuePos;
var plDefShortestDistances;
var plDefWayPoints;
var plDefPathQueueX;
var plDefPathQueueY;
var plDefX;
var plDefY;
var plDefStandStillCounter;
//}
//{ Food - f
function fFood(x, y, isGood, type = "t") {
	this.x = x;
	this.y = y;
	this.isGood = isGood;
	this.type = type;
	if (this.isGood) {
		this.colorRed = 0;
		this.colorGreen = 255;
	} else {
		this.colorRed = 255;
		this.colorGreen = 0;
	}
	this.colorBlue = 0;
	this.id = foodIDCounter;
	foodIDCounter++;
}
var foodIDCounter;
//}
//{ RunnerRNG - rng
const rngSOUTH = 0;
const rngWEST = 1;
const rngEAST = 2;
function rngRunnerRNG(forcedMovements) {
	this.forcedMovements = forcedMovements;
	this.forcedMovementsIndex = 0;

	this.rollMovement = function() {
		if (this.forcedMovements.length > this.forcedMovementsIndex) {
			let movement = this.forcedMovements.charAt(this.forcedMovementsIndex++);
			if (movement === "s") {
				return rngSOUTH;
			}
			if (movement === "w") {
				return rngWEST;
			}
			if (movement === "e") {
				return rngEAST;
			}
		}
		let rnd = Math.floor(Math.random() * 6);
		if (rnd < 4) {
			return rngSOUTH;
		}
		if (rnd === 4) {
			return rngWEST;
		}
		return rngEAST;
	}
}
//}
//{ Runner - ru
function ruInit(sniffDistance) {
	ruSniffDistance = sniffDistance;
}
function ruRunner(x, y, runnerRNG, isWave10, id) {
	this.x = x;
	this.y = y;
	this.destinationX = x;
	this.destinationY = y;
	this.cycleTick = 1;
	this.targetState = 0;
	this.foodTarget = null;
	this.blughhhhCountdown = 0;
	this.standStillCounter = 0;
	this.despawnCountdown = -1;
	this.isDying = false;
	this.runnerRNG = runnerRNG;
	this.isWave10 = isWave10;
	this.id = id;

	this.tick = function() {
		if (++this.cycleTick > 10) {
			this.cycleTick = 1;
		}
		++this.standStillCounter;
		if (this.despawnCountdown !== -1) {
			if (--this.despawnCountdown === 0) {
				baRunnersToRemove.push(this);
				if (!this.isDying) {
					--baRunnersAlive;
				} else {
					if (baIsNearEastTrap(this.x, this.y)) {
						if (eastTrapState > 0) --eastTrapState;
					}
					if (baIsNearWestTrap(this.x, this.y)) {
						if (westTrapState > 0) --westTrapState;
					}
				}
			}
		} else {
			if (!this.isDying) {
				switch(this.cycleTick) {
					case 1:
						this.doTick1();
						break;
					case 2:
						this.doTick2Or5();
						break;
					case 3:
						this.doTick3();
						break;
					case 4:
						this.doTick4();
						break;
					case 5:
						this.doTick2Or5();
						break;
					case 6:
						this.doTick6();
						break;
					case 7:
					case 8:
					case 9:
					case 10:
						this.doTick7To10();
						break;
				}
			}
			if (this.isDying) {
				if (this.standStillCounter > 2) {
					++baRunnersKilled;
					--baRunnersAlive;
					this.print("Urghhh!");
					this.despawnCountdown = 2;
				}
			}
		}
	}

	this.doMovement = function() { // TODO: Doesn't consider diagonal movement block flags
		let startX = this.x;
		if (this.destinationX > startX) {
			if (!baTileBlocksPenance(startX + 1, this.y) && mCanMoveEast(startX, this.y)) {
				++this.x;
				this.standStillCounter = 0;
			}
		} else if (this.destinationX < startX && !baTileBlocksPenance(startX - 1, this.y) && mCanMoveWest(startX, this.y)) {
			--this.x;
			this.standStillCounter = 0;
		}
		if (this.destinationY > this.y) {
			if (!baTileBlocksPenance(startX, this.y + 1) && !baTileBlocksPenance(this.x, this.y + 1) && mCanMoveNorth(startX, this.y) && mCanMoveNorth(this.x, this.y)) {
				++this.y;
				this.standStillCounter = 0;
			}
		} else if (this.destinationY < this.y && !baTileBlocksPenance(startX, this.y - 1) && !baTileBlocksPenance(this.x, this.y - 1) && mCanMoveSouth(startX, this.y) && mCanMoveSouth(this.x, this.y)) {
			--this.y;
			this.standStillCounter = 0;
		}
	}

	this.tryTargetFood = function() {
		let xZone = this.x >> 3;
		let yZone = this.y >> 3;
		let firstFoodFound = null;
		let endXZone = Math.max(xZone - 1 , 0);
		let endYZone = Math.max(yZone - 1, 0);
		for (let x = Math.min(xZone + 1, mItemZonesWidth - 1); x >= endXZone; --x) {
			for (let y = Math.min(yZone + 1, mItemZonesHeight - 1); y >= endYZone; --y) {
				let itemZone = mGetItemZone(x, y);
				for (let foodIndex = itemZone.length - 1; foodIndex >= 0; --foodIndex) {
					let food = itemZone[foodIndex];
					if (!mHasLineOfSight(this.x, this.y, food.x, food.y)) {
						continue;
					}
					if (firstFoodFound === null) {
						firstFoodFound = food;
					}
					if (Math.max(Math.abs(this.x - food.x), Math.abs(this.y - food.y)) <= ruSniffDistance) {
						this.foodTarget = firstFoodFound;
						this.destinationX = firstFoodFound.x;
						this.destinationY = firstFoodFound.y;
						this.targetState = 0;
						return;
					}
				}
			}
		}
	}

	this.tryEatAndCheckTarget = function() {
		// experimental retarget mechanism on multikill tick
		/*
        if (baTickCounter > 1 && baTickCounter % 10 === 4) { // multikill tick
            this.tryTargetFood();
        }*/
		if (this.foodTarget !== null) {
			let itemZone = mGetItemZone(this.foodTarget.x >>> 3, this.foodTarget.y >>> 3);
			let foodIndex = itemZone.indexOf(this.foodTarget);
			if (foodIndex === -1) {
				this.foodTarget = null;
				this.targetState = 0;
				return true;
			} else if (this.x === this.foodTarget.x && this.y === this.foodTarget.y) {
				let targetID = this.foodTarget.id;
				if (this.foodTarget.isGood) {
					this.print("Chomp, chomp.");

					if (baIsNearEastTrap(this.x, this.y)) {
						if (eastTrapState > 0 || requireRepairs === "no") {
							this.isDying = true;
						}
					} else if (baIsNearWestTrap(this.x, this.y)) {
						if (westTrapState > 0 || requireRepairs === "no") {
							this.isDying = true;
						}
					}
				} else {
					this.print("Blughhhh.");
					this.blughhhhCountdown = 3;
					this.targetState = 0;
					if (this.cycleTick > 5) {
						this.cycleTick -= 5;
					}
					this.setDestinationBlughhhh();
				}
				itemZone.splice(foodIndex, 1);
				for (let i = 0; i < foodList.length; i++) {
					if (foodList[i].id === targetID) {
						foodList.splice(i, 1);
					}
				}
				return true;
			}
		}
		return false;
	}

	this.cancelDestination = function() {
		this.destinationX = this.x;
		this.destinationY = this.y;
	}

	this.setDestinationBlughhhh = function() {
		this.destinationX = this.x;
		if (this.isWave10) {
			this.destinationY = baEAST_TRAP_Y - 4;
		} else {
			this.destinationY = baEAST_TRAP_Y + 4;
		}
	}

	this.setDestinationRandomWalk = function() {
		if (this.x <= 27) { // TODO: These same for wave 10?
			if (this.y === 8 || this.y === 9) {
				this.destinationX = 30;
				this.destinationY = 8;
				return;
			} else if (this.x === 25 && this.y === 7) {
				this.destinationX = 26;
				this.destinationY = 8;
				return;
			}
		} else if (this.x <= 32) {
			if (this.y <= 8) {
				this.destinationX = 30;
				this.destinationY = 6;
				return;
			}
		} else if (this.y === 7 || this.y === 8){
			this.destinationX = 31;
			this.destinationY = 8;
			return;
		}
		let direction = this.runnerRNG.rollMovement();
		if (direction === rngSOUTH) {
			this.destinationX = this.x;
			this.destinationY = this.y - 5;
		} else if (direction === rngWEST) {
			this.destinationX = this.x - 5;
			if (this.destinationX < baWEST_TRAP_X - 1) { // TODO: Same for wave 10?
				this.destinationX = baWEST_TRAP_X - 1;
			}
			this.destinationY = this.y;
		} else {
			this.destinationX = this.x + 5;
			if (this.isWave10) {
				if (this.destinationX > baEAST_TRAP_X - 1) {
					this.destinationX = baEAST_TRAP_X - 1;
				}
			} else if (this.destinationX > baEAST_TRAP_X) {
				this.destinationX = baEAST_TRAP_X;
			}
			this.destinationY = this.y;
		}
	}

	this.doTick1 = function() {
		if (this.y === 6) {
			this.despawnCountdown = 3;
			this.print("Raaa!");
			return;
		}
		if (this.blughhhhCountdown > 0) {
			--this.blughhhhCountdown;
		} else {
			++this.targetState;
			if (this.targetState > 3) {
				this.targetState = 1;
			}
		}
		let ateOrTargetGone = this.tryEatAndCheckTarget();
		if (this.blughhhhCountdown === 0 && this.foodTarget === null) { // Could make this line same as tick 6 without any difference?
			this.cancelDestination();
		}
		if (!ateOrTargetGone) {
			this.doMovement();
		}
	}

	this.doTick2Or5 = function() {
		if (this.targetState === 2) {
			this.tryTargetFood();
		}
		this.doTick7To10();
	}

	this.doTick3 = function() {
		if (this.targetState === 3) {
			this.tryTargetFood();
		}
		this.doTick7To10();
	}

	this.doTick4 = function() {
		if (this.targetState === 1) {
			this.tryTargetFood();
		}
		this.doTick7To10();
	}

	this.doTick6 = function() {
		if (this.y === 6) {
			this.despawnCountdown = 3;
			this.print("Raaa!");
			return;
		}
		if (this.blughhhhCountdown > 0) {
			--this.blughhhhCountdown;
		}
		if (this.targetState === 3) {
			this.tryTargetFood();
		}
		let ateOrTargetGone = this.tryEatAndCheckTarget();
		if (this.blughhhhCountdown === 0 && (this.foodTarget === null || ateOrTargetGone)) {
			this.setDestinationRandomWalk();
		}
		if (!ateOrTargetGone) {
			this.doMovement();
		}
	}

	this.doTick7To10 = function() {
		let ateOrTargetGone = this.tryEatAndCheckTarget();
		if (!ateOrTargetGone) {
			this.doMovement();
		}
	}

	this.print = function(string) {
		console.log(baTickCounter + ": Runner " + this.id + ": " + string);
	}

}
var ruSniffDistance;
//}
//{ BaArena - ba
const baWEST_TRAP_X = 15;
const baWEST_TRAP_Y = 25;
const baEAST_TRAP_X = 45;
const baEAST_TRAP_Y = 26;
const baWAVE1_RUNNER_SPAWN_X = 36;
const baWAVE1_RUNNER_SPAWN_Y = 39;
const baWAVE10_RUNNER_SPAWN_X = 42;
const baWAVE10_RUNNER_SPAWN_Y = 38;
const baWAVE1_DEFENDER_SPAWN_X = 33;
const baWAVE1_DEFENDER_SPAWN_Y = 8;
const baWAVE10_DEFENDER_SPAWN_X = 28;
const baWAVE10_DEFENDER_SPAWN_Y = 8;

const WAVE1_NW_LOGS_X = 28;
const WAVE1_NW_LOGS_Y = 39;
const WAVE10_NW_LOGS_X = 29;
const WAVE10_NW_LOGS_Y = 39;
const WAVE1_SE_LOGS_X = 29;
const WAVE1_SE_LOGS_Y = 38;
const WAVE10_SE_LOGS_X = 30;
const WAVE10_SE_LOGS_Y = 38;
const HAMMER_X = 32;
const HAMMER_Y = 34;
function baInit(maxRunnersAlive, totalRunners, runnerMovements) {
	baRunners = [];
	baRunnersToRemove = [];
	baTickCounter = 0;
	baRunnersAlive = 0;
	baRunnersKilled = 0;
	baMaxRunnersAlive = maxRunnersAlive;
	baTotalRunners = totalRunners;
	numCrackers = 9;
	numTofu = 9;
	numWorms = 9;
	numLogs = 0;
	hasHammer = false;
	eastTrapState = 2;
	westTrapState = 2;
	currDefFood = "t";
	northwestLogsState = true;
	southeastLogsState = true;
	hammerState = true;
	baCollectorX = -1;
	baRunnerMovements = runnerMovements;
	baRunnerMovementsIndex = 0;
	baCurrentRunnerId = 1;
	simTickCountSpan.innerHTML = baTickCounter;
	currDefFoodSpan.innerHTML = currDefFood;
	isPaused = false;
	foodIDCounter = 0;
	foodList = [];
}
function baTick() {
	++baTickCounter;
	baRunnersToRemove.length = 0;
	for (let i = 0; i < baRunners.length; ++i) {
		baRunners[i].tick();
	}
	for (let i = 0; i < baRunnersToRemove.length; ++i) {
		let runner = baRunnersToRemove[i];
		let index = baRunners.indexOf(runner);
		baRunners.splice(index, 1);
	}
	// hammer and logs respawn
	if (baTickCounter > 1 && baTickCounter % 10 === 1) {
		northwestLogsState = true;
		southeastLogsState = true;
		hammerState = true;
	}
	// currDefFood changes
	if (baTickCounter > 2 && baTickCounter % 50 === 2) {
		if (currDefFood === "t") {
			if (Math.random() < 0.5) {
				currDefFood = "c";
			} else {
				currDefFood = "w";
			}
		} else if (currDefFood === "c") {
			if (Math.random() < 0.5) {
				currDefFood = "w";
			} else {
				currDefFood = "t";
			}
		} else {
			if (Math.random() < 0.5) {
				currDefFood = "t";
			} else {
				currDefFood = "c";
			}
		}
		currDefFoodSpan.innerHTML = currDefFood;
	}
	if (baTickCounter > 1 && baTickCounter % 10 === 1 && baRunnersAlive < baMaxRunnersAlive && baRunnersKilled + baRunnersAlive < baTotalRunners) {
		let movements;
		if (baRunnerMovements.length > baRunnerMovementsIndex) {
			movements = baRunnerMovements[baRunnerMovementsIndex++];
		} else {
			movements = "";
		}
		if (mCurrentMap === mWAVE_1_TO_9) {
			baRunners.push(new ruRunner(baWAVE1_RUNNER_SPAWN_X, baWAVE1_RUNNER_SPAWN_Y, new rngRunnerRNG(movements), false, baCurrentRunnerId++));
		} else {
			baRunners.push(new ruRunner(baWAVE10_RUNNER_SPAWN_X, baWAVE10_RUNNER_SPAWN_Y, new rngRunnerRNG(movements), true, baCurrentRunnerId++));
		}
		++baRunnersAlive;
	}
	simTickCountSpan.innerHTML = baTickCounter;
}
function baDrawOverlays() { 
	if (mCurrentMap !== mWAVE_1_TO_9 && mCurrentMap !== mWAVE10) {
		return;
	}
	rSetDrawColor(240, 10, 10, 220);
	if (mCurrentMap === mWAVE_1_TO_9) {
		rrOutline(18, 37);
	} else {
		rrOutline(18, 38);
	}
	rrOutline(24, 39);
	rrFill(33, 6);
	rSetDrawColor(10, 10, 240, 220);
	if (mCurrentMap === mWAVE_1_TO_9) {
		rrOutline(36, 39);
	} else {
		rrOutline(42, 38);
	}
	rrFill(34, 6);
	rSetDrawColor(10, 240, 10, 220);
	if (mCurrentMap === mWAVE_1_TO_9) {
		rrOutline(42, 37);
	} else {
		rrOutline(36, 39);
	}
	rrFill(35, 6);
	rSetDrawColor(240, 240, 10, 220);
	rrFill(36, 6);
}
function baDrawDetails() {
	if (mCurrentMap !== mWAVE_1_TO_9 && mCurrentMap !== mWAVE10) {
		return;
	}
	rSetDrawColor(160, 82, 45, 255); // logs and trap color
	rrCone(40, 32);
	rrCone(40, 31);
	rrCone(41, 32);
	rrCone(41, 31);
	rrCone(43, 31);
	rrCone(36, 34);
	rrCone(36, 35);
	rrCone(37, 34);
	rrCone(37, 35);
	rrCone(39, 36);
	rrCone(43, 22);
	rrCone(43, 23);
	rrCone(44, 22);
	rrCone(44, 23);
	rrCone(45, 24);
	if (mCurrentMap === mWAVE_1_TO_9) {
		if (southeastLogsState) {
			rrFillItem(WAVE1_SE_LOGS_X, WAVE1_SE_LOGS_Y); // se logs 1-9
		}
		if (northwestLogsState) {
			rrFillItem(WAVE1_NW_LOGS_X, WAVE1_NW_LOGS_Y); // nw logs 1-9
		}
	} else {
		if (southeastLogsState) {
			rrFillItem(WAVE10_SE_LOGS_X, WAVE10_SE_LOGS_Y); // se logs 10
		}
		if (northwestLogsState) {
			rrFillItem(WAVE10_NW_LOGS_X, WAVE10_NW_LOGS_Y); // nw logs 10
		}
	}
	if (eastTrapState < 1) {
		rSetDrawColor(255, 0, 0, 255);
	} else if (eastTrapState === 1) {
		rSetDrawColor(255, 140, 0, 255);
	}
	rrOutline(45, 26); // e trap
	rSetDrawColor(160, 82, 45, 255);
	if (westTrapState < 1) {
		rSetDrawColor(255, 0, 0, 255);
	} else if (westTrapState === 1) {
		rSetDrawColor(255, 140, 0, 255);
	}
	rrOutline(15, 25); // w trap
	rSetDrawColor(160, 82, 45, 255);
	if (mCurrentMap === mWAVE10) {
		rrOutlineBig(27, 20, 8, 8); // queen thing
	}
	rSetDrawColor(127, 127, 127, 255); // hammer color
	if (hammerState) {
		rrFillItem(HAMMER_X, HAMMER_Y); // hammer
	}
}
function baDrawEntities() {
	rSetDrawColor(10, 10, 240, 127);
	for (let i = 0; i < baRunners.length; ++i) {
		rrFill(baRunners[i].x, baRunners[i].y);
	}
	if (baCollectorX !== -1) {
		rSetDrawColor(240, 240, 10, 200);
		rrFill(baCollectorX, baCollectorY);
	}
}
function baIsNearTrap(x, y) {
	return (Math.abs(x - baEAST_TRAP_X) < 2 && Math.abs(y - baEAST_TRAP_Y) < 2) || (Math.abs(x - baWEST_TRAP_X) < 2 && Math.abs(y - baWEST_TRAP_Y) < 2);
}
function baIsNearEastTrap(x, y) {
	return (Math.abs(x - baEAST_TRAP_X) < 2 && Math.abs(y - baEAST_TRAP_Y) < 2);
}
function baIsNearWestTrap(x, y) {
	return (Math.abs(x - baWEST_TRAP_X) < 2 && Math.abs(y - baWEST_TRAP_Y) < 2);
}
function isInEastRepairRange(x, y) {
	return Math.abs(x - baEAST_TRAP_X) + Math.abs(y - baEAST_TRAP_Y) < 2;
}
function isInWestRepairRange(x, y) {
	return Math.abs(x - baWEST_TRAP_X) + Math.abs(y - baWEST_TRAP_Y) < 2;
}
function baTileBlocksPenance(x, y) {
	if (x === plDefX && y === plDefY) {
		return true;
	}
	if (x === baCollectorX && y === baCollectorY) {
		return true;
	}
	if (y === 22) {
		if (x >= 20 && x <= 22) {
			return true;
		}
		if (mCurrentMap === mWAVE_1_TO_9 && x >= 39 && x <= 41) {
			return true;
		}
	} else if (x === 46 && y >= 9 && y <= 12) {
		return true;
	} else if (mCurrentMap === mWAVE_1_TO_9 && x === 27 && y === 24) {
		return true;
	}
	return false;
}
var baRunners;
var baRunnersToRemove;
var baTickCounter;
var baRunnersAlive;
var baRunnersKilled;
var baTotalRunners;
var baMaxRunnersAlive;
var baCollectorX;
var baCollectorY;
var baRunnerMovements;
var baRunnerMovementsIndex;
var baCurrentRunnerId;
//}
//{ Map - m
const mWAVE_1_TO_9 = [16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097156,2097154,2097154,2097154,2097154,2228480,2228480,2228480,2228480,2097154,2097154,2097154,2097154,2097153,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2097156,2097408,96,2097440,2097440,32,0,0,0,0,131360,131360,131360,131376,2097408,2097153,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,131328,131328,131328,2228480,2097156,2097154,2097154,2097408,64,0,2097408,2097408,0,0,0,0,0,0,0,0,0,16,2097408,2097154,2097154,2097154,2097154,2097154,2097154,2097154,2097154,2097154,2097153,2228480,2228480,2228480,2228480,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,131328,2228480,2097156,2097154,2097154,2097408,352,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,32,32,32,32,32,32,131362,131386,2228608,131328,0,0,2228480,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,131328,131328,2097156,2097408,96,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,32,0,0,0,0,0,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2228480,131328,131328,2097160,192,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2228480,131328,2097156,2097408,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,131328,2097156,2097408,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0,131328,131328,0,0,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2228480,2228480,2097160,192,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131360,131368,2097538,0,131328,0,0,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2228480,2228480,2228480,2097160,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131368,2097280,0,131328,131328,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2228480,2228480,2097156,2097408,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131336,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,131328,2097156,2097408,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2228480,2097160,192,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2228480,2097160,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2228480,2097160,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2228480,2097160,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,131328,2097160,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,131328,2097160,128,0,0,0,0,0,4104,65664,0,4104,65664,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4104,65664,0,4104,65664,0,0,0,0,0,8,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,131328,2097160,129,0,0,0,0,0,5130,65664,0,4104,66690,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5130,65664,0,4104,66690,0,0,0,0,0,8,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2228480,2097168,2097408,0,0,0,0,4104,2310560,0,0,0,2249000,65664,0,0,0,0,0,0,0,0,0,0,0,0,4104,2310560,0,0,0,2249000,65664,0,0,0,0,8,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2228480,2228480,2228480,2097160,128,0,0,0,4104,65664,0,0,0,4104,65664,0,0,0,0,0,0,0,0,0,0,0,0,4104,65664,0,0,0,4104,65664,0,0,0,0,12,2097280,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2228480,2228480,2097156,2097408,0,0,0,0,4104,65664,0,262144,131328,4104,65664,0,0,0,0,0,0,0,0,0,0,0,0,4104,65664,0,262144,131328,4104,65664,0,0,0,4,2097408,2097216,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2228480,2097156,2097408,64,0,0,0,0,4104,2295170,1026,1026,1026,2233610,65664,0,0,0,0,0,0,0,0,0,0,0,0,4104,2295170,1026,1026,1026,2233610,65664,0,0,0,2097408,2097216,2228480,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2228480,2097160,192,0,0,0,0,0,0,16416,16416,16416,16416,16416,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16416,16416,16416,16416,16416,0,0,0,8,2097280,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2228480,2097160,129,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2097280,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2228480,2097168,2097408,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2097408,2097153,2228480,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2228480,2228480,2097168,2097408,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,2097280,2228480,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2228480,2228480,2228480,2097160,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,2097280,2228480,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2228480,131328,2097160,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2097408,2097216,2228480,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2228480,2097160,129,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2097408,2097216,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2228480,2097168,2097408,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2097280,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2228480,2228480,2097168,2097408,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,2097280,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097160,129,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2097408,2097216,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097168,2097408,1,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0,0,4,2097408,2097216,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097168,2228480,2228480,2228480,2097184,2097184,2097408,1,0,0,2,2,0,0,0,0,0,2,2,0,0,4,2097408,2097184,2097184,2228480,2228480,2228480,2097216,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097168,2228480,2228480,2228480,2097184,2097184,2097408,3,2,6,2097408,2097184,2097184,2228480,2228480,2228480,2097216,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097168,2097184,2097184,2097184,2097216,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,16777216,16777216,16777216,16777216,16777216,16777216,16777216,16777216];
const mWAVE10 = [2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2097156,2097154,2097154,2097154,2097154,2228480,2228480,2228480,2228480,2097154,2097154,2097154,2097154,2228481,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097156,2097408,96,2097440,2097440,32,0,0,0,0,131360,131360,131360,131376,2097408,2228481,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,131328,131328,131328,2228480,2097156,2097154,2097154,2097408,64,0,2097408,2097408,0,0,0,0,0,0,0,0,0,16,2097408,2097154,2097154,2097154,2097154,2097154,2097154,2097154,2097154,2097154,2097153,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,131328,2097156,2097154,2097154,2097154,2097408,352,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,32,32,32,32,32,32,131362,131386,2097280,131328,0,0,131328,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097156,2097408,96,131360,32,0,0,0,131328,0,0,0,0,0,0,0,0,0,0,0,0,131328,0,131328,131328,0,131328,0,0,0,0,32,32,0,0,0,0,0,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,131328,2097156,2097408,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131328,0,0,0,0,0,0,0,0,0,0,0,0,131328,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2097160,192,131328,0,0,0,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097156,2097408,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,0,0,0,0,0,2,2,0,131328,131328,0,0,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097156,2097408,64,0,0,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,0,131360,131368,2097538,0,131328,0,0,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097160,192,131328,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131368,2097280,0,131328,131328,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2097160,128,131328,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131336,2097280,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2097156,2097408,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,0,0,0,0,2097408,2097153,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097156,2097408,131392,0,0,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,0,0,131328,131328,0,16,2097408,2097153,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097156,2097408,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131328,0,0,24,2097280,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097160,192,131328,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,8,2097280,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097160,128,131328,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,0,0,0,0,2097408,2097153,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097160,128,0,0,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,2097408,2097153,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097156,2097408,131328,0,0,0,0,0,0,0,0,4104,65664,0,4104,65664,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131328,24,2097280,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097160,192,0,0,0,0,0,0,0,0,0,5130,65664,0,4104,66690,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131328,8,2097280,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097160,128,131328,131328,0,0,0,0,0,0,4104,2179488,0,0,0,2117928,65664,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,0,0,131340,2097280,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097160,129,131328,131328,0,0,0,0,0,0,4104,65664,0,0,0,4104,65664,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2097408,2097216,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097168,2097408,0,0,0,131328,0,0,0,0,4104,65664,0,262144,131328,4104,65664,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,8,2097280,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097160,128,0,131328,0,0,0,0,0,4104,2164098,1026,1026,1026,2102538,65664,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,2097280,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097160,129,0,0,0,0,0,0,0,0,16416,16416,16416,16416,16416,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131328,0,2097408,2097216,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097168,2097408,1,0,0,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131328,12,2097280,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097168,2097408,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,0,0,0,4,2097408,2097216,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2097160,129,0,131328,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,0,2097408,2097216,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097168,2097408,1,131328,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131328,0,12,2097280,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097168,2097408,0,0,0,0,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,131328,4,2097408,2097216,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2097160,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2097408,2097216,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097160,129,0,131328,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131328,12,2097280,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2097168,2097408,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2097408,2097216,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2097168,2097408,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2097408,2097216,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2097168,2097408,1,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0,0,4,2097408,2097216,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2097168,2228480,2228480,2228480,2097184,2097184,2097408,1,0,0,2,2,0,0,0,0,0,2,2,0,0,4,2097408,2097184,2097184,2228480,2228480,2228480,2097216,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097168,2228480,2228480,2228480,2097184,2097184,2097408,3,2,6,2097408,2097184,2097184,2228480,2228480,2228480,2097216,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097168,2097184,2097184,2097184,2097216,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2228480,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152];
const mLOS_FULL_MASK = 0x20000;
const mLOS_EAST_MASK = 0x1000;
const mLOS_WEST_MASK = 0x10000;
const mLOS_NORTH_MASK = 0x400;
const mLOS_SOUTH_MASK = 0x4000;
const mMOVE_FULL_MASK = 0x100 | 0x200000 | 0x40000 | 0x1000000; // 0x100 for objects, 0x200000 for unwalkable tiles such as water etc, 0x40000 is very rare but BA cannon has it. 0x1000000 is not confirmed to block move but outside ba arena 1-9.
const mMOVE_EAST_MASK = 0x8;
const mMOVE_WEST_MASK = 0x80;
const mMOVE_NORTH_MASK = 0x2;
const mMOVE_SOUTH_MASK = 0x20;
function mInit(map, widthTiles, heightTiles) {
	mCurrentMap = map;
	mWidthTiles = widthTiles;
	mHeightTiles = heightTiles;
	mResetMap();
}
function mResetMap() {
	mItemZones = [];
	mItemZonesWidth = 1 + ((mWidthTiles - 1) >> 3);
	mItemZonesHeight = 1 + ((mHeightTiles - 1) >> 3);
	for (let xZone = 0; xZone < mItemZonesWidth; ++xZone) {
		for (let yZone = 0; yZone < mItemZonesHeight; ++yZone) {
			mItemZones[xZone + mItemZonesWidth*yZone] = [];
		}
	}
}
function mAddItem(item) {
	mGetItemZone(item.x >>> 3, item.y >>> 3).push(item);
	foodList.push(item);
}
function mGetItemZone(xZone, yZone) {
	return mItemZones[xZone + mItemZonesWidth*yZone];
}
function mGetTileFlag(x, y) {
	return mCurrentMap[x + y*mWidthTiles];
}
function mCanMoveEast(x, y) {
	return (mGetTileFlag(x + 1, y) & (mMOVE_WEST_MASK | mMOVE_FULL_MASK)) === 0;
}
function mCanMoveWest(x, y) {
	return (mGetTileFlag(x - 1, y) & (mMOVE_EAST_MASK | mMOVE_FULL_MASK)) === 0;
}
function mCanMoveNorth(x, y) {
	return (mGetTileFlag(x, y + 1) & (mMOVE_SOUTH_MASK | mMOVE_FULL_MASK)) === 0;
}
function mCanMoveSouth(x, y) {
	return (mGetTileFlag(x, y - 1) & (mMOVE_NORTH_MASK | mMOVE_FULL_MASK)) === 0;
}
function mDrawGrid() {
	for (var xTile = 0; xTile < mWidthTiles; ++xTile) {
		if (xTile % 8 == 7) {
			rSetDrawColor(0, 0, 0, 72);
		} else {
			rSetDrawColor(0, 0, 0, 48);
		}
		rrEastLineBig(xTile, 0, mHeightTiles);
	}
	for (var yTile = 0; yTile < mHeightTiles; ++yTile) {
		if (yTile % 8 == 7) {
			rSetDrawColor(0, 0, 0, 72);
		} else {
			rSetDrawColor(0, 0, 0, 48);
		}
		rrNorthLineBig(0, yTile, mWidthTiles);
	}
}
function mDrawItems() {
	let endI = mItemZones.length;
	for (let i = 0; i < endI; ++i) {
		let itemZone = mItemZones[i];
		let endJ = itemZone.length;
		for (let j = 0; j < endJ; ++j) {
			let item = itemZone[j];
			rSetDrawColor(item.colorRed, item.colorGreen, item.colorBlue, 127);
			rrFillItem(item.x, item.y);
		}
	}
}
function mDrawMap() {
	rSetDrawColor(206, 183, 117, 255);
	rClear();
	for (let y = 0; y < mHeightTiles; ++y) {	
		for (let x = 0; x < mWidthTiles; ++x) {
			let tileFlag = mGetTileFlag(x, y);
			if ((tileFlag & mLOS_FULL_MASK) !== 0) {
				rSetDrawColor(0, 0, 0, 255);
				rrFillOpaque(x, y);
			} else  {
				if ((tileFlag & mMOVE_FULL_MASK) !== 0) {
					rSetDrawColor(127, 127, 127, 255);
					rrFillOpaque(x, y);
				}
				if ((tileFlag & mLOS_EAST_MASK) !== 0) {
					rSetDrawColor(0, 0, 0, 255);
					rrEastLine(x, y);
				} else if ((tileFlag & mMOVE_EAST_MASK) !== 0) {
					rSetDrawColor(127, 127, 127, 255);
					rrEastLine(x, y);
				}
				if ((tileFlag & mLOS_WEST_MASK) !== 0) {
					rSetDrawColor(0, 0, 0, 255);
					rrWestLine(x, y);
				} else if ((tileFlag & mMOVE_WEST_MASK) !== 0) {
					rSetDrawColor(127, 127, 127, 255);
					rrWestLine(x, y);
				}
				if ((tileFlag & mLOS_NORTH_MASK) !== 0) {
					rSetDrawColor(0, 0, 0, 255);
					rrNorthLine(x, y);
				} else if ((tileFlag & mMOVE_NORTH_MASK) !== 0) {
					rSetDrawColor(127, 127, 127, 255);
					rrNorthLine(x, y);
				}
				if ((tileFlag & mLOS_SOUTH_MASK) !== 0) {
					rSetDrawColor(0, 0, 0, 255);
					rrSouthLine(x, y);
				} else if ((tileFlag & mMOVE_SOUTH_MASK) !== 0) {
					rSetDrawColor(127, 127, 127, 255);
					rrSouthLine(x, y);
				}
			}
		}
	}
}
function mHasLineOfSight(x1, y1, x2, y2) {
	let dx = x2 - x1;
	let dxAbs = Math.abs(dx);
	let dy = y2 - y1;
	let dyAbs = Math.abs(dy);
	if (dxAbs > dyAbs) {
		let xTile = x1;
		let y = y1 << 16;
		let slope = Math.trunc((dy << 16) / dxAbs);
		let xInc;
		let xMask;
		if (dx > 0) {
			xInc = 1;
			xMask = mLOS_WEST_MASK | mLOS_FULL_MASK;
		} else {
			xInc = -1;
			xMask = mLOS_EAST_MASK | mLOS_FULL_MASK;
		}
		let yMask;
		y += 0x8000;
		if (dy < 0) {
			y -= 1;
			yMask = mLOS_NORTH_MASK | mLOS_FULL_MASK;
		} else {
			yMask = mLOS_SOUTH_MASK | mLOS_FULL_MASK;
		}
		while (xTile !== x2) {
			xTile += xInc;
			let yTile = y >>> 16;
			if ((mGetTileFlag(xTile, yTile) & xMask) !== 0) {
				return false;
			}
			y += slope;
			let newYTile = y >>> 16;
			if (newYTile !== yTile && (mGetTileFlag(xTile, newYTile) & yMask) !== 0) {
				return false;
			}
		}
	} else {
		let yTile = y1;
		let x = x1 << 16;
		let slope = Math.trunc((dx << 16) / dyAbs);
		let yInc;
		let yMask;
		if (dy > 0) {
			yInc = 1;
			yMask = mLOS_SOUTH_MASK | mLOS_FULL_MASK;
		} else {
			yInc = -1;
			yMask = mLOS_NORTH_MASK | mLOS_FULL_MASK;
		}
		let xMask;
		x += 0x8000;
		if (dx < 0) {
			x -= 1;
			xMask = mLOS_EAST_MASK | mLOS_FULL_MASK;
		} else {
			xMask = mLOS_WEST_MASK | mLOS_FULL_MASK;
		}
		while (yTile !== y2) {
			yTile += yInc;
			let xTile = x >>> 16;
			if ((mGetTileFlag(xTile, yTile) & yMask) !== 0) {
				return false;
			}
			x += slope;
			let newXTile = x >>> 16;
			if (newXTile !== xTile && (mGetTileFlag(newXTile, yTile) & xMask) !== 0) {
				return false;
			}
		}
	}
	return true;
}
var mCurrentMap;
var mWidthTiles;
var mHeightTiles;
var mItemZones;
var mItemZonesWidth;
var mItemZonesHeight;
//}
//{ RsRenderer - rr
function rrInit(tileSize) {
	rrTileSize = tileSize;
}
function rrSetTileSize(size) {
	rrTileSize = size;
}
function rrSetSize(widthTiles, heightTiles) {
	rrWidthTiles = widthTiles;
	rrHeightTiles = heightTiles;
	rResizeCanvas(rrTileSize*rrWidthTiles, rrTileSize*rrHeightTiles);
}
function rrFillOpaque(x, y) {
	rSetFilledRect(x*rrTileSize, y*rrTileSize, rrTileSize, rrTileSize);
}
function rrFill(x, y) {
	rDrawFilledRect(x*rrTileSize, y*rrTileSize, rrTileSize, rrTileSize);
}
function rrFillBig(x, y, width, height) {
	rDrawFilledRect(x*rrTileSize, y*rrTileSize, width*rrTileSize, height*rrTileSize);
}
function rrOutline(x, y) {
	rDrawOutlinedRect(x*rrTileSize, y*rrTileSize, rrTileSize, rrTileSize);
}
function rrOutlineBig(x, y, width, height) {
	rDrawOutlinedRect(x*rrTileSize, y*rrTileSize, rrTileSize*width, rrTileSize*height);
}
function rrWestLine(x, y) {
	rDrawVerticalLine(x*rrTileSize, y*rrTileSize, rrTileSize);
}
function rrWestLineBig(x, y, length) {
	rDrawHorizontalLine(x*rrTileSize, y*rrTileSize, rrTileSize*length)
}
function rrEastLine(x, y) {
	rDrawVerticalLine((x + 1)*rrTileSize - 1, y*rrTileSize, rrTileSize);
}
function rrEastLineBig(x, y, length) {
	rDrawVerticalLine((x + 1)*rrTileSize - 1, y*rrTileSize, rrTileSize*length);
}
function rrSouthLine(x, y) {
	rDrawHorizontalLine(x*rrTileSize, y*rrTileSize, rrTileSize);
}
function rrSouthLineBig(x, y, length) {
	rDrawHorizontalLine(x*rrTileSize, y*rrTileSize, rrTileSize*length);
}
function rrNorthLine(x, y) {
	rDrawHorizontalLine(x*rrTileSize, (y + 1)*rrTileSize - 1, rrTileSize);
}
function rrNorthLineBig(x, y, length) {
	rDrawHorizontalLine(x*rrTileSize, (y + 1)*rrTileSize - 1, rrTileSize*length);
}
function rrCone(x, y) {
	rDrawCone(x*rrTileSize, y*rrTileSize, rrTileSize);
}
function rrFillItem(x, y) {
	let padding = rrTileSize >>> 2;
	let size = rrTileSize - 2*padding;
	rDrawFilledRect(x*rrTileSize + padding, y*rrTileSize + padding, size, size);
}
var rrTileSize;
//}
//{ Renderer - r
const rPIXEL_ALPHA = 255 << 24;
function rInit(canvas, width, height) {
	rCanvas = canvas;
	rContext = canvas.getContext("2d");
	rResizeCanvas(width, height);
	rSetDrawColor(255, 255, 255, 255);
}
function rResizeCanvas(width, height) {
	rCanvas.width = width;
	rCanvas.height = height;
	rCanvasWidth = width;
	rCanvasHeight = height;
	rCanvasYFixOffset = (rCanvasHeight - 1)*rCanvasWidth;
	rImageData = rContext.createImageData(width, height);
	rPixels = new ArrayBuffer(rImageData.data.length);
	rPixels8 = new Uint8ClampedArray(rPixels);
	rPixels32 = new Uint32Array(rPixels);
}
function rSetDrawColor(r, g, b, a) {
	rDrawColorRB = r | (b << 16);
	rDrawColorG = rPIXEL_ALPHA | (g << 8);
	rDrawColor = rDrawColorRB | rDrawColorG;
	rDrawColorA = a + 1;
}
function rClear() {
	let endI = rPixels32.length;
	for (let i = 0; i < endI; ++i) {
		rPixels32[i] = rDrawColor;
	}
}
function rPresent() {
	rImageData.data.set(rPixels8);
	rContext.putImageData(rImageData, 0, 0);
}
function rDrawPixel(i) {
	let color = rPixels32[i];
	let oldRB = color & 0xFF00FF;
	let oldAG = color & 0xFF00FF00;
	let rb = oldRB + (rDrawColorA*(rDrawColorRB - oldRB) >> 8) & 0xFF00FF;
	let g = oldAG + (rDrawColorA*(rDrawColorG - oldAG) >> 8) & 0xFF00FF00;
	rPixels32[i] = rb | g;
}
function rDrawHorizontalLine(x, y, length) {
	let i = rXYToI(x, y)
	let endI = i + length;
	for (; i < endI; ++i) {
		rDrawPixel(i);
	}
}
function rDrawVerticalLine(x, y, length) {
	let i = rXYToI(x, y);
	let endI = i - length*rCanvasWidth;
	for (; i > endI; i -= rCanvasWidth) {
		rDrawPixel(i);
	}
}
function rSetFilledRect(x, y, width, height) {
	let i = rXYToI(x, y);
	let rowDelta = width + rCanvasWidth;
	let endYI = i - height*rCanvasWidth;
	while (i > endYI) {
		let endXI = i + width;
		for (; i < endXI; ++i) {
			rPixels32[i] = rDrawColor;
		}
		i -= rowDelta;
	}
}
function rDrawFilledRect(x, y, width, height) {
	let i = rXYToI(x, y);
	let rowDelta = width + rCanvasWidth;
	let endYI = i - height*rCanvasWidth;
	while (i > endYI) {
		let endXI = i + width;
		for (; i < endXI; ++i) {
			rDrawPixel(i);
		}
		i -= rowDelta;
	}
}
function rDrawOutlinedRect(x, y, width, height) {
	rDrawHorizontalLine(x, y, width);
	rDrawHorizontalLine(x, y + height - 1, width);
	rDrawVerticalLine(x, y + 1, height - 2);
	rDrawVerticalLine(x + width - 1, y + 1, height - 2);
}
function rDrawCone(x, y, width) { // Not optimised to use i yet
	let lastX = x + width - 1;
	let endI = (width >>> 1) + (width & 1);
	for (let i = 0; i < endI; ++i) {
		rDrawPixel(rXYToI(x + i, y));
		rDrawPixel(rXYToI(lastX - i, y));
		++y;
	}
}
function rXYToI(x, y) {
	return rCanvasYFixOffset + x - y*rCanvasWidth;
}
var rCanvas;
var rCanvasWidth;
var rCanvasHeight;
var rCanvasYFixOffset;
var rContext;
var rImageData;
var rPixels;
var rPixels8;
var rPixels32;
var rDrawColor;
var rDrawColorRB;
var rDrawColorG;
var rDrawColorA;
//}

// HERE BEGINS THE SAVE/LOAD CODE

var savebaRunners;
var savebaRunnersToRemove;
var savebaTickCounter;
var savebaRunnersAlive;
var savebaRunnersKilled;
var savebaMaxRunnersAlive;
var savebaTotalRunners;
var savenumCrackers;
var savenumTofu;
var savenumWorms;
var savenumLogs;
var savehasHammer;
var saveeastTrapState;
var savewestTrapState;
var savecurrDefFood;
var savenorthwestLogsState;
var savesoutheastLogsState;
var savehammerState;
var savebaCollectorX;
var savebaRunnerMovements;
var savebaRunnerMovementsIndex;
var savebaCurrentRunnerId;

var saveisPaused;
var savebaCollectorY;

// WEIRD STUFF
var savesimTickCountSpaninnerHTML; // ???
var savecurrDefFoodSpaninnerHTML; // ???
//var savesimTickTimerId; //  000000000000  use currently running sim tick timer
//var savesimMovementsInput; // 000000000000  movements are saved/loaded via baRunnerMovements
//var savesimStartStopButton; // 000000000000  use existing startstop button
//var savesimWaveSelect; // 000000000000  wave info is saved/loaded via other variables
//var savesimDefLevelSelect; // 0000000000000 passed via rusniffdistance
//var savesimToggleRepair; // 000000000000 use currently running togglerepair
//var savesimIsRunning; // 0000000000000 use currently running simisrunning
//var savesimTickDurationInput; // 0000000000000  use currently running sim tick duration
// NO MORE WEIRD STUFF

var saverequireRepairs;

var savepickingUpFood; // "t", "c", "w", "n"
var savepickingUpLogs; // true/false
var savepickingUpHammer; // true/false
var saverepairTicksRemaining; // 0-5

var saveplDefPathQueuePos;
var saveplDefShortestDistances;
var saveplDefWayPoints;
var saveplDefPathQueueX;
var saveplDefPathQueueY;
var saveplDefX;
var saveplDefY;
var saveplDefStandStillCounter;

var savemCurrentMap;
var savemWidthTiles;
var savemHeightTiles;
var savemItemZones;
var savemItemZonesWidth;
var savemItemZonesHeight;

var saverrTileSize;

var saverCanvas;
var saverCanvasWidth;
var saverCanvasHeight;
var saverCanvasYFixOffset;
var saverContext;
var saverImageData;
var saverPixels;
var saverPixels8;
var saverPixels32;
var saverDrawColor;
var saverDrawColorRB;
var saverDrawColorG;
var saverDrawColorA;

var saveruSniffDistance;

function deepCopy(obj) {
	return JSON.parse(JSON.stringify(obj));
}

function otherDeepCopy(obj) {
	return obj.map(a => Object.assign({}, a))
}

const v8 = require('v8');

function v8deepCopy(obj) {
	return v8.deserialize(v8.serialize(obj));
}

const clonedeep = require('lodash.clonedeep');

function lodashDeepCopy(obj) {
	return clonedeep(obj);
}

function saveGameState() {
	isPaused = true; // pause before saving

	// WEIRD STUFF
	savesimTickCountSpaninnerHTML = simTickCountSpan.innerHTML;
	savecurrDefFoodSpaninnerHTML = currDefFoodSpan.innerHTML;
	// NO MORE WEIRD STUFF

	savebaRunners = otherDeepCopy(baRunners);
	savebaRunnersToRemove = otherDeepCopy(baRunnersToRemove);
	savebaTickCounter = baTickCounter;
	savebaRunnersAlive = baRunnersAlive;
	savebaRunnersKilled = baRunnersKilled;
	savebaMaxRunnersAlive = baMaxRunnersAlive;
	savebaTotalRunners = baTotalRunners;
	savenumCrackers = numCrackers;
	savenumTofu = numTofu;
	savenumWorms = numWorms;
	savenumLogs = numLogs;
	savehasHammer = hasHammer;
	saveeastTrapState = eastTrapState;
	savewestTrapState = westTrapState;
	savecurrDefFood = currDefFood;
	savenorthwestLogsState = northwestLogsState;
	savesoutheastLogsState = southeastLogsState;
	savehammerState = hammerState;
	savebaCollectorX = baCollectorX;
	savebaRunnerMovements = deepCopy(baRunnerMovements);
	savebaRunnerMovementsIndex = baRunnerMovementsIndex;
	savebaCurrentRunnerId = baCurrentRunnerId;
	saveisPaused = isPaused;
	savebaCollectorY = baCollectorY;

	//saverequireRepairs = requireRepairs;
	savepickingUpFood = pickingUpFood;
	savepickingUpHammer = pickingUpHammer;
	saverepairTicksRemaining = repairTicksRemaining;

	saveplDefPathQueuePos = plDefPathQueuePos;
	saveplDefShortestDistances = deepCopy(plDefShortestDistances);
	saveplDefWayPoints = deepCopy(plDefWayPoints);
	saveplDefPathQueueX = deepCopy(plDefPathQueueX);
	saveplDefPathQueueY = deepCopy(plDefPathQueueY);
	saveplDefX = plDefX;
	saveplDefY = plDefY;
	saveplDefStandStillCounter = plDefStandStillCounter;

	savemCurrentMap = mCurrentMap;
	savemWidthTiles = mWidthTiles;
	savemHeightTiles = mHeightTiles;
	savemItemZones = deepCopy(mItemZones);
	savemItemZonesWidth = mItemZonesWidth;
	savemItemZonesHeight = mItemZonesHeight;

	saverrTileSize = rrTileSize;

	saverCanvas = rCanvas;
	saverCanvasWidth = rCanvasWidth;
	saverCanvasHeight = rCanvasHeight;
	saverCanvasYFixOffset = rCanvasYFixOffset;
	saverContext = rContext;
	saverImageData = rImageData;
	saverPixels = rPixels;
	saverPixels8 = rPixels8;
	saverPixels32 = rPixels32;
	saverDrawColor = rDrawColor;
	saverDrawColorRB = rDrawColorRB;
	saverDrawColorG = rDrawColorG;
	saverDrawColorA = rDrawColorA;

	saveruSniffDistance = ruSniffDistance;
}

function loadGameState() {
	isPaused = true;

	// WEIRD STUFF
	simTickCountSpan.innerHTML = savesimTickCountSpaninnerHTML;
	currDefFoodSpan.innerHTML = savecurrDefFoodSpaninnerHTML;
	// NO MORE WEIRD STUFF

	baRunners = otherDeepCopy(savebaRunners);
	baRunnersToRemove = otherDeepCopy(savebaRunnersToRemove);
	baTickCounter = savebaTickCounter;
	baRunnersAlive = savebaRunnersAlive;
	baRunnersKilled = savebaRunnersKilled;
	baMaxRunnersAlive = savebaMaxRunnersAlive;
	baTotalRunners = savebaTotalRunners;
	numCrackers = savenumCrackers;
	numTofu = savenumTofu;
	numWorms = savenumWorms;
	numLogs = savenumLogs;
	hasHammer = savehasHammer;
	eastTrapState = saveeastTrapState;
	westTrapState = savewestTrapState;
	currDefFood = savecurrDefFood;
	northwestLogsState = savenorthwestLogsState;
	southeastLogsState = savesoutheastLogsState;
	hammerState = savehammerState;
	baCollectorX = savebaCollectorX;
	baRunnerMovements = deepCopy(savebaRunnerMovements);
	baRunnerMovementsIndex = savebaRunnerMovementsIndex;
	baCurrentRunnerId = savebaCurrentRunnerId;
	isPaused = saveisPaused;
	baCollectorY = savebaCollectorY;

	//requireRepairs = saverequireRepairs;
	pickingUpFood = savepickingUpFood;
	pickingUpHammer = savepickingUpHammer;
	repairTicksRemaining = saverepairTicksRemaining;

	plDefPathQueuePos = saveplDefPathQueuePos;
	plDefShortestDistances = deepCopy(saveplDefShortestDistances);
	plDefWayPoints = deepCopy(saveplDefWayPoints);
	plDefPathQueueX = deepCopy(saveplDefPathQueueX);
	plDefPathQueueY = deepCopy(saveplDefPathQueueY);
	plDefX = saveplDefX;
	plDefY = saveplDefY;
	plDefStandStillCounter = saveplDefStandStillCounter;

	mCurrentMap = savemCurrentMap;
	mWidthTiles = savemWidthTiles;
	mHeightTiles = savemHeightTiles;
	mItemZones = deepCopy(savemItemZones);
	mItemZonesWidth = savemItemZonesWidth;
	mItemZonesHeight = savemItemZonesHeight;

	rrTileSize = saverrTileSize;

	rCanvas = saverCanvas;
	rCanvasWidth = saverCanvasWidth;
	rCanvasHeight = saverCanvasHeight;
	rCanvasYFixOffset = saverCanvasYFixOffset;
	rContext = saverContext;
	rImageData = saverImageData;
	rPixels = saverPixels;
	rPixels8 = saverPixels8;
	rPixels32 = saverPixels32;
	rDrawColor = saverDrawColor;
	rDrawColorRB = saverDrawColorRB;
	rDrawColorG = saverDrawColorG;
	rDrawColorA = saverDrawColorA;

	ruSniffDistance = saveruSniffDistance;

	for (let i = 0; i < baRunners.length; i++) {
		let thisRunner = baRunners[i];
		if (thisRunner.foodTarget !== null) {
			let thisRunnerFoodID = thisRunner.foodTarget.id;
			for (let j = 0; j < mItemZones.length; j++) {
				let itemZone = mItemZones[j];
				for (let k = 0; k < itemZone.length; k++) {
					let thisFood = itemZone[k];
					if (thisFood.id === thisRunnerFoodID) {
						thisRunner.foodTarget = thisFood;
					}
				}
			}
		}
	}

	requireRepairs = simToggleRepair.value;
	pauseSL = simTogglePauseSL.value;
	infiniteFood = simToggleInfiniteFood.value;
	logHammerToRepair = simToggleLogHammerToRepair.value;

	simDraw();
}

// RL HERE

function getFoodLocations() {
	let result = new Array(81);

	for (let i = 0; i < numTofu; i++) {
		result[i * 3] = 1; // good=1 bad=0
		result[i * 3 + 1] = -1; // x
		result[i * 3 + 2] = -1; // y
	}
	let numTofuOnGround = 0;
	for (let i = 0; i < foodList.length; i++) {
		let food = foodList[i];
		if (food.type === "t") {
			result[numTofu * 3 + i * 3] = food.isGood ? 1 : 0;
			result[numTofu * 3 + i * 3 + 1] = food.x;
			result[numTofu * 3 + i * 3 + 2] = food.y;
			numTofuOnGround += 1;
		}
	}
	for (let i = numTofuOnGround + numTofu; i < 9; i++) {
		result[i * 3] = -1; // good=1 bad=0
		result[i * 3 + 1] = -1; // x
		result[i * 3 + 2] = -1; // y
	}

	for (let i = 0; i < numCrackers; i++) {
		result[27 + i * 3] = 1; // good=1 bad=0
		result[27 + i * 3 + 1] = -1; // x
		result[27 + i * 3 + 2] = -1; // y
	}
	let numCrackersOnGround = 0;
	for (let i = 0; i < foodList.length; i++) {
		let food = foodList[i];
		if (food.type === "c") {
			result[27 + numCrackers * 3 + i * 3] = food.isGood ? 1 : 0;
			result[27 + numCrackers * 3 + i * 3 + 1] = food.x;
			result[27 + numCrackers * 3 + i * 3 + 2] = food.y;
			numCrackersOnGround += 1;
		}
	}
	for (let i = numCrackersOnGround + numCrackers; i < 9; i++) {
		result[27 + i * 3] = -1; // good=1 bad=0
		result[27 + i * 3 + 1] = -1; // x
		result[27 + i * 3 + 2] = -1; // y
	}

	for (let i = 0; i < numWorms; i++) {
		result[54 + i * 3] = 1; // good=1 bad=0
		result[54 + i * 3 + 1] = -1; // x
		result[54 + i * 3 + 2] = -1; // y
	}
	let numWormsOnGround = 0;
	for (let i = 0; i < foodList.length; i++) {
		let food = foodList[i];
		if (food.type === "w") {
			result[54 + numWorms * 3 + i * 3] = food.isGood ? 1 : 0;
			result[54 + numWorms * 3 + i * 3 + 1] = food.x;
			result[54 + numWorms * 3 + i * 3 + 2] = food.y;
			numWormsOnGround += 1;
		}
	}
	for (let i = numWormsOnGround + numWorms; i < 9; i++) {
		result[54 + i * 3] = -1; // good=1 bad=0
		result[54 + i * 3 + 1] = -1; // x
		result[54 + i * 3 + 2] = -1; // y
	}

	return result;
}

function getRunnerInfo() {

	let result = new Array(10 * 2);

	for (let i = 0; i < baRunners.length; i++) {
		let runner = baRunners[i];
		result[i * 10] = runner.isDying ? 1 : 0;

		if (runner.foodTarget !== null) {
			result[i * 10 + 1] = runner.foodTarget.id;
			result[i * 10 + 2] = runner.foodTarget.x;
			result[i * 10 + 3] = runner.foodTarget.y;
			result[i * 10 + 4] = runner.foodTarget.isGood ? 1 : 0;
		} else {
			result[i * 10 + 1] = -1;
			result[i * 10 + 2] = -1;
			result[i * 10 + 3] = -1;
			result[i * 10 + 4] = -1;
		}

		result[i * 10 + 5] = runner.x;
		result[i * 10 + 6] = runner.y;
		result[i * 10 + 7] = runner.standStillCounter;
		result[i * 10 + 8] = runner.despawnCountdown;
		result[i * 10 + 9] = runner.blughhhhCountdown;
	}

	for (let i = baRunners.length; i < 2; i++) {
		result[i * 10] = -1;
		result[i * 10 + 1] = -1;
		result[i * 10 + 2] = -1;
		result[i * 10 + 3] = -1;
		result[i * 10 + 4] = -1;
		result[i * 10 + 5] = -1;
		result[i * 10 + 6] = -1;
		result[i * 10 + 7] = -1;
		result[i * 10 + 8] = -1;
		result[i * 10 + 9] = -1;
	}

	return result;
}

function rlTick() {
	agent.forward();
	baTick();
	plDefTick();
	simDraw();
	agent.backward();
}