'use strict';// RL.JS

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

//{ Simulation - sim
function simInit() {
    mInit(mWAVE_1_TO_9, 64, 48);
    ruInit(5);
    simReset();

    //agent = new RunnerWave1Agent();
    agent = new Wave1TestAgent();

    env = agent;
    spec = {};
    spec.update = 'qlearn'; // qlearn | sarsa
    spec.gamma = 0.9; // discount factor, [0, 1)
    spec.epsilon = 0.8;//0.2 // initial epsilon for epsilon-greedy policy, [0, 1)
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
    baInit(0, 0, "");
    plDefInit(-1, 0);
}
//const fs = require('fs');
function simSaveAgent() {
    let jsonData = JSON.stringify(agent.brain.toJSON());

    require('fs').writeFileSync("agent.json", jsonData); // node.js
}
function simLoadAgent() {
    let jsonData = require('fs').readFileSync("agent.json");

    agent.brain.fromJSON(jsonData); // node.js
}
function simStartStop() {
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
        let maxRunnersAlive = 0;
        let totalRunners = 0;
        let wave = 1;
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
        simTickTimerId = setInterval(rlTick, Number(0));
        //simTick();
        //simTickTimerId = setInterval(simTick, Number(simTickDurationInput.value)); // tick time in milliseconds (set to 600 for real)
    }
}
function simParseMovementsInput(movementsInput) {
    let movements = movementsInput.split("-");
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

function simTick() {
    if (!isPaused) {
        baTick();
        plDefTick();
    }
}

var simTickTimerId;
var simMovementsInput;
var simStartStopButton;
var simSaveAgentButton;
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
        //console.log(baTickCounter + ": Runner " + this.id + ": " + string);
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

var rrTileSize;
//}
//{ Renderer - r
const rPIXEL_ALPHA = 255 << 24;

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

function saveGameState() {
    isPaused = true; // pause before saving

    // WEIRD STUFF
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

    requireRepairs = "yes";
    pauseSL = "no";
    infiniteFood = "no";
    logHammerToRepair = "yes";

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
    agent.backward();

    if (baTickCounter % 100000 === 0) {
        console.log(baTickCounter);
    }
    if (baTickCounter % 500000 === 0) {
        simSaveAgent();
        console.log("agent saved!");
    }
    if (baTickCounter === 5000000) {
        clearInterval(simTickTimerId);
    }
}

// MAIN FUNCTION

function mainFunction() {
    simInit();
    simStartStop();
}

//mainFunction();

// RUNNER ALGORITHM

function solveForThreeTiles() {
    var tiles = [];

    var x1min = baEAST_TRAP_X - 10;
    var x1max = baEAST_TRAP_X + 10;
    var y1min = baEAST_TRAP_Y;
    var y1max = baEAST_TRAP_Y + 10;
    var x2min = baEAST_TRAP_X - 10;
    var x2max = baEAST_TRAP_X + 10;
    var y2min = baEAST_TRAP_Y;
    var y2max = baEAST_TRAP_Y + 15;
    var x3min = baEAST_TRAP_X - 1;
    var x3max = baEAST_TRAP_X + 1;
    var y3min = baEAST_TRAP_Y - 1;
    var y3max = baEAST_TRAP_Y + 1;

    for (let x1 = x1min; x1 <= x1max; x1++) {

        for (let y1 = y1min; y1 <= y1max; y1++) {

            var tile1 = [x1, y1];

            if (isValidTile(tile1) && isInRange(x1, y1, baEAST_TRAP_X, baEAST_TRAP_Y + 1, 4)) {

                for (let x2 = x2min; x2 <= x2max; x2++) {

                    for (let y2 = y2min; y2 <= y2max; y2++) {

                        var tile2 = [x2, y2];

                        if (isValidTile(tile2) && isInRange(x1, y1, x2, y2, 5)) {

                            for (let x3 = x3min; x3 <= x3max; x3++) {

                                for (let y3 = y3min; y3 <= y3max; y3++) {

                                    var tile3 = [x3, y3];

                                    if (isValidTile(tile3) && isInRange(x3, y3, baEAST_TRAP_X, baEAST_TRAP_Y, 1) && (getWalkDistance(x3, y3, x2, y2) === 10)) {

                                        if (checkThreeTiles(tile1, tile2, tile3)) {
                                            tiles.push([tile1, tile2, tile3]);
                                        }

                                    }

                                }
                            }
                        }
                    }
                }

            }
        }
    }

    return tiles;
}

function solveForFourTiles() {
    var tiles = [];

    var x1min = baEAST_TRAP_X - 10;
    var x1max = baEAST_TRAP_X + 10;
    var y1min = baEAST_TRAP_Y;
    var y1max = baEAST_TRAP_Y + 10;
    var x2min = baEAST_TRAP_X - 10;
    var x2max = baEAST_TRAP_X + 10;
    var y2min = baEAST_TRAP_Y;
    var y2max = baEAST_TRAP_Y + 15;
    var x3min = baEAST_TRAP_X - 10;
    var x3max = baEAST_TRAP_X + 10;
    var y3min = baEAST_TRAP_Y;
    var y3max = baEAST_TRAP_Y + 15;
    var x4min = baEAST_TRAP_X - 1;
    var x4max = baEAST_TRAP_X + 1;
    var y4min = baEAST_TRAP_Y - 1;
    var y4max = baEAST_TRAP_Y + 1;

    for (let x1 = x1min; x1 <= x1max; x1++) {

        for (let y1 = y1min; y1 <= y1max; y1++) {

            var tile1 = [x1, y1];

            if (isValidTile(tile1) && isInRange(x1, y1, baEAST_TRAP_X, baEAST_TRAP_Y + 1, 4)) {

                for (let x2 = x2min; x2 <= x2max; x2++) {

                    for (let y2 = y2min; y2 <= y2max; y2++) {

                        var tile2 = [x2, y2];

                        if (isValidTile(tile2) && isInRange(x1, y1, x2, y2, 5)) {

                            for (let x3 = x3min; x3 <= x3max; x3++) {

                                for (let y3 = y3min; y3 <= y3max; y3++) {

                                    var tile3 = [x3, y3];

                                    if (isValidTile(tile3) && isInRange(x2, y2, x3, y3, 5)) {

                                        for (let x4 = x4min; x4 <= x4max; x4++) {

                                            for (let y4 = y4min; y4 <= y4max; y4++) {

                                                var tile4 = [x4, y4];

                                                if (isValidTile(tile4) && isInRange(x4, y4, baEAST_TRAP_X, baEAST_TRAP_Y, 1) && (getWalkDistance(x4, y4, x3, y3) === 10)) {

                                                    if (checkFourTiles(tile1, tile2, tile3, tile4)) {
                                                        tiles.push([tile1, tile2, tile3, tile4]);
                                                    }

                                                }

                                            }

                                        }

                                    }

                                }
                            }
                        }
                    }
                }

            }
        }
    }

    return tiles;
}

function solveForFiveTiles() {
    var tiles = [];

    var x1min = baEAST_TRAP_X - 10;
    var x1max = baEAST_TRAP_X + 10;
    var y1min = baEAST_TRAP_Y;
    var y1max = baEAST_TRAP_Y + 10;
    var x2min = baEAST_TRAP_X - 10;
    var x2max = baEAST_TRAP_X + 10;
    var y2min = baEAST_TRAP_Y;
    var y2max = baEAST_TRAP_Y + 15;
    var x3min = baEAST_TRAP_X - 10;
    var x3max = baEAST_TRAP_X + 10;
    var y3min = baEAST_TRAP_Y;
    var y3max = baEAST_TRAP_Y + 15;
    var x4min = baEAST_TRAP_X - 10;
    var x4max = baEAST_TRAP_X + 10;
    var y4min = baEAST_TRAP_Y;
    var y4max = baEAST_TRAP_Y + 15;
    var x5min = baEAST_TRAP_X - 1;
    var x5max = baEAST_TRAP_X + 1;
    var y5min = baEAST_TRAP_Y - 1;
    var y5max = baEAST_TRAP_Y + 1;

    for (let x1 = x1min; x1 <= x1max; x1++) {

        for (let y1 = y1min; y1 <= y1max; y1++) {

            var tile1 = [x1, y1];

            if (isValidTile(tile1) && isInRange(x1, y1, baEAST_TRAP_X, baEAST_TRAP_Y + 1, 4)) {

                for (let x2 = x2min; x2 <= x2max; x2++) {

                    for (let y2 = y2min; y2 <= y2max; y2++) {

                        var tile2 = [x2, y2];

                        if (isValidTile(tile2) && isInRange(x1, y1, x2, y2, 5)) {

                            for (let x3 = x3min; x3 <= x3max; x3++) {

                                for (let y3 = y3min; y3 <= y3max; y3++) {

                                    var tile3 = [x3, y3];

                                    if (isValidTile(tile3) && isInRange(x2, y2, x3, y3, 5)) {

                                        for (let x4 = x4min; x4 <= x4max; x4++) {

                                            for (let y4 = y4min; y4 <= y4max; y4++) {

                                                var tile4 = [x4, y4];

                                                if (isValidTile(tile4) && isInRange(x3, y3, x4, y4, 5)) {

                                                    for (let x5 = x5min; x5 <= x5max; x5++) {

                                                        for (let y5 = y5min; y5 <= y5max; y5++) {

                                                            var tile5 = [x5, y5];

                                                            if (isValidTile(tile5) && isInRange(x5, y5, baEAST_TRAP_X, baEAST_TRAP_Y, 1) && isInRange(x5, y5, x4, y4, 5)) {

                                                                if (checkFiveTiles(tile1, tile2, tile3, tile4, tile5)) {
                                                                    tiles.push([tile1, tile2, tile3, tile4, tile5]);
                                                                }

                                                            }

                                                        }

                                                    }

                                                }

                                            }

                                        }

                                    }

                                }
                            }
                        }
                    }
                }

            }
        }
    }

    return tiles;
}

function appendMovement(currentMovements, movementOptions,) {
    var result = [];

    for (let i = 0; i < currentMovements.length; i++) {
        for (let j = 0; j < movementOptions.length; j++) {
            result.push(currentMovements[i] + '-' + movementOptions[j]);
        }
    }

    return result;
}

function isRunnerOnStack() {
    for (let i = 0; i < baRunners.length; i++) {
        var thisRunner = baRunners[i];
        if ((thisRunner.x === baEAST_TRAP_X) && (thisRunner.y === baEAST_TRAP_Y + 1)) {
            return true;
        }
    }
    return false;
}

function checkFiveTiles(tile1, tile2, tile3, tile4, tile5) {

    var movementsInputList = ['ss-ss'];
    var movementOptions = ['ss', 'sw', 'se', 'ws', 'ww', 'we', 'es', 'ew', 'ee'];

    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);

    //var movementsInputList = ['s-s-s', 's-s-w', 's-s-e', 's-w-s', 's-w-w', 's-w-e', 's-e-s', 's-e-w', 's-e-e', 'w-s-s', 'w-s-w', 'w-s-e', 'w-w-s', 'w-w-w', 'w-w-e', 'w-e-s', 'w-e-w', 'w-e-e', 'e-s-s', 'e-s-w', 'e-s-e', 'e-w-s', 'e-w-w', 'e-w-e', 'e-e-s', 'e-e-w', 'e-e-e'];

    for (let movementsIndex = 0; movementsIndex < movementsInputList.length; movementsIndex++) {
        simInit();

        mResetMap();
        simIsRunning = false;
        baInit(0, 0, "");
        plDefInit(-1, 0);

        let movements = simParseMovementsInput(movementsInputList[movementsIndex]);
        //let movements = simParseMovementsInput('ss-ss-ss-' + movementsInputList[movementsIndex]);
        simIsRunning = true;
        let maxRunnersAlive = 5;
        let totalRunners = 6;
        let wave = 7;
        baInit(maxRunnersAlive, totalRunners, movements);
        plDefInit(baWAVE1_DEFENDER_SPAWN_X, baWAVE1_DEFENDER_SPAWN_Y);

        var firstRunner = false;
        var fourthDropTick = 110;

        for (let i = 0; i < 110; i++) {
            simTick();

            if (baTickCounter === 24) { // trap food
                mAddItem(new fFood(baEAST_TRAP_X, baEAST_TRAP_Y + 1, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X, baEAST_TRAP_Y + 1, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X, baEAST_TRAP_Y + 1, true, "t"));
            } else if (baTickCounter === 27) { // trail food
                mAddItem(new fFood(baEAST_TRAP_X - 6, baEAST_TRAP_Y + 5, false, "w"));
            } else if (baTickCounter === 29) { // main stack
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
            } else if (!firstRunner && isRunnerOnStack()) {
                firstRunner = true;
                plDefX = baEAST_TRAP_X;
                plDefY = baEAST_TRAP_Y + 1;
            } else if (baTickCounter === 43) { // move defender onto trap food to block runners from eating, after first starts eating
                //plDefX = baEAST_TRAP_X;
                //plDefY = baEAST_TRAP_Y + 1;
            } else if (baTickCounter === 61) { // move off trap food on tick 61
                plDefPathfind(tile1[0], tile1[1]);
            } else if ((plDefX === tile1[0]) && (plDefY === tile1[1])) { // drop bait food asap, then move to tile2
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                plDefPathfind(tile2[0], tile2[1]);
            } else if ((plDefX === tile2[0]) && (plDefY === tile2[1])) {
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                plDefPathfind(tile3[0], tile3[1]);
            } else if ((plDefX === tile3[0]) && (plDefY === tile3[1])) {
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                plDefPathfind(tile4[0], tile4[1]);
            } else if ((plDefX === tile4[0]) && (plDefY === tile4[1])) { // drop bait food asap, then teleport to tile3 to avoid bumping
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                plDefX = baWAVE1_DEFENDER_SPAWN_X;
                plDefY = baWAVE1_DEFENDER_SPAWN_Y;
                fourthDropTick = baTickCounter;
            } else if (baTickCounter === fourthDropTick + 5) { // should take 5 ticks to get to tile3 from tile2, then teleport away to not block
                plDefX = tile5[0];
                plDefY = tile5[1];
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                plDefX = baWAVE1_DEFENDER_SPAWN_X;
                plDefY = baWAVE1_DEFENDER_SPAWN_Y;
            }

            if ((baTickCounter === 100) && baRunnersKilled > 5) {
                return false;
            }
        }

        if (baRunnersKilled < 6) {
            console.log(tile1 + ', ' + tile2 + ', ' + tile3 + ', ' + tile4 + ', ' + tile5 + ' failed with ' + baRunnersKilled + ' runners killed on pattern ' + movements);
            return false;
        }
    }

    console.log(tile1 + ', ' + tile2 + ', ' + tile3 + ', ' + tile4 + ', ' + tile5 + ' succeeded');
    return true;
}

function checkFourTiles(tile1, tile2, tile3, tile4) {

    var movementsInputList = ['ss'];
    var movementOptions = ['ss', 'sw', 'se', 'ws', 'ww', 'we', 'es', 'ew', 'ee'];

    movementsInputList = appendMovement(movementsInputList, ['ss', 'se', 'ws', 'we', 'es', 'ew', 'ee']);
    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);

    //var movementsInputList = ['s-s-s', 's-s-w', 's-s-e', 's-w-s', 's-w-w', 's-w-e', 's-e-s', 's-e-w', 's-e-e', 'w-s-s', 'w-s-w', 'w-s-e', 'w-w-s', 'w-w-w', 'w-w-e', 'w-e-s', 'w-e-w', 'w-e-e', 'e-s-s', 'e-s-w', 'e-s-e', 'e-w-s', 'e-w-w', 'e-w-e', 'e-e-s', 'e-e-w', 'e-e-e'];

    for (let movementsIndex = 0; movementsIndex < movementsInputList.length; movementsIndex++) {
        simInit();

        mResetMap();
        simIsRunning = false;
        baInit(0, 0, "");
        plDefInit(-1, 0);

        let movements = simParseMovementsInput(movementsInputList[movementsIndex]);
        //let movements = simParseMovementsInput('ss-ss-ss-' + movementsInputList[movementsIndex]);
        simIsRunning = true;
        let maxRunnersAlive = 5;
        let totalRunners = 6;
        let wave = 7;
        baInit(maxRunnersAlive, totalRunners, movements);
        plDefInit(baWAVE1_DEFENDER_SPAWN_X, baWAVE1_DEFENDER_SPAWN_Y);

        var firstRunner = false;
        var thirdDropTick = 110;

        for (let i = 0; i < 110; i++) {
            simTick();

            if (baTickCounter === 24) { // trap food
                mAddItem(new fFood(baEAST_TRAP_X, baEAST_TRAP_Y + 1, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X, baEAST_TRAP_Y + 1, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X, baEAST_TRAP_Y + 1, true, "t"));
            } else if (baTickCounter === 27) { // trail food
                mAddItem(new fFood(baEAST_TRAP_X - 6, baEAST_TRAP_Y + 5, false, "w"));
            } else if (baTickCounter === 29) { // main stack
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
            } else if (!firstRunner && isRunnerOnStack()) {
                firstRunner = true;
                plDefX = baEAST_TRAP_X;
                plDefY = baEAST_TRAP_Y + 1;
            } else if (baTickCounter === 43) { // move defender onto trap food to block runners from eating, after first starts eating
                //plDefX = baEAST_TRAP_X;
                //plDefY = baEAST_TRAP_Y + 1;
            } else if (baTickCounter === 61) { // move off trap food on tick 61
                plDefPathfind(tile1[0], tile1[1]);
            } else if ((plDefX === tile1[0]) && (plDefY === tile1[1])) { // drop bait food asap, then move to tile2
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                plDefPathfind(tile2[0], tile2[1]);
            } else if ((plDefX === tile2[0]) && (plDefY === tile2[1])) {
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                plDefPathfind(tile3[0], tile3[1]);
            } else if ((plDefX === tile3[0]) && (plDefY === tile3[1])) { // drop bait food asap, then teleport to tile3 to avoid bumping
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                plDefX = baWAVE1_DEFENDER_SPAWN_X;
                plDefY = baWAVE1_DEFENDER_SPAWN_Y;
                thirdDropTick = baTickCounter;
            } else if (baTickCounter === thirdDropTick + 5) { // should take 5 ticks to get to tile3 from tile2, then teleport away to not block
                plDefX = tile4[0];
                plDefY = tile4[1];
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                mAddItem(new fFood(plDefX, plDefY, true, "c"));
                plDefX = baWAVE1_DEFENDER_SPAWN_X;
                plDefY = baWAVE1_DEFENDER_SPAWN_Y;
            }

            if ((baTickCounter === 100) && baRunnersKilled > 5) {
                return false;
            }
        }

        if (baRunnersKilled < 6) {
            console.log(tile1 + ', ' + tile2 + ', ' + tile3 + ', ' + tile4 + ' failed with ' + baRunnersKilled + ' runners killed on pattern ' + movements);
            return false;
        }
    }

    console.log(tile1 + ', ' + tile2 + ', ' + tile3 + ', ' + tile4 + ' succeeded');
    return true;
}

function checkThreeTiles(tile1, tile2, tile3) {

    var movementsInputList = ['se'];
    var movementOptions = ['ss', 'sw', 'se', 'ws', 'ww', 'we', 'es', 'ew', 'ee'];

    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);
    movementsInputList = appendMovement(movementsInputList, movementOptions);

    //var movementsInputList = ['s-s-s', 's-s-w', 's-s-e', 's-w-s', 's-w-w', 's-w-e', 's-e-s', 's-e-w', 's-e-e', 'w-s-s', 'w-s-w', 'w-s-e', 'w-w-s', 'w-w-w', 'w-w-e', 'w-e-s', 'w-e-w', 'w-e-e', 'e-s-s', 'e-s-w', 'e-s-e', 'e-w-s', 'e-w-w', 'e-w-e', 'e-e-s', 'e-e-w', 'e-e-e'];

    for (let movementsIndex = 0; movementsIndex < movementsInputList.length; movementsIndex++) {
        simInit();

        mResetMap();
        simIsRunning = false;
        baInit(0, 0, "");
        plDefInit(-1, 0);

        let movements = simParseMovementsInput(movementsInputList[movementsIndex]);
        //let movements = simParseMovementsInput('ss-ss-ss-' + movementsInputList[movementsIndex]);
        simIsRunning = true;
        let maxRunnersAlive = 5;
        let totalRunners = 6;
        let wave = 7;
        baInit(maxRunnersAlive, totalRunners, movements);
        plDefInit(baWAVE1_DEFENDER_SPAWN_X, baWAVE1_DEFENDER_SPAWN_Y);

        var firstRunner = false;
        var secondDropTick = 100;

        for (let i = 0; i < 100; i++) {
            simTick();

            if (baTickCounter === 24) { // trap food
                mAddItem(new fFood(baEAST_TRAP_X, baEAST_TRAP_Y + 1, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X, baEAST_TRAP_Y + 1, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X, baEAST_TRAP_Y + 1, true, "t"));
            } else if (baTickCounter === 27) { // trail food
                mAddItem(new fFood(baEAST_TRAP_X - 6, baEAST_TRAP_Y + 5, false, "w"));
            } else if (baTickCounter === 29) { // main stack
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
                mAddItem(new fFood(baEAST_TRAP_X - 10, baEAST_TRAP_Y + 9, true, "t"));
            } else if (!firstRunner && isRunnerOnStack()) {
                firstRunner = true;
                plDefX = baEAST_TRAP_X;
                plDefY = baEAST_TRAP_Y + 1;
            } else if (baTickCounter === 43) { // move defender onto trap food to block runners from eating, after first starts eating
                //plDefX = baEAST_TRAP_X;
                //plDefY = baEAST_TRAP_Y + 1;
            } else if (baTickCounter === 61) { // move off trap food on tick 61
                plDefPathfind(tile1[0], tile1[1]);
            } else if ((plDefX === tile1[0]) && (plDefY === tile1[1])) { // drop bait food asap, then move to tile2
                mAddItem(new fFood(plDefX, plDefY, true, "w"));
                plDefPathfind(tile2[0], tile2[1]);
            } else if ((plDefX === tile2[0]) && (plDefY === tile2[1])) { // drop bait food asap, then teleport to tile3 to avoid bumping
                mAddItem(new fFood(plDefX, plDefY, true, "w"));
                plDefX = baWAVE1_DEFENDER_SPAWN_X;
                plDefY = baWAVE1_DEFENDER_SPAWN_Y;
                secondDropTick = baTickCounter;
            } else if (baTickCounter === secondDropTick + 5) { // should take 5 ticks to get to tile3 from tile2, then teleport away to not block
                plDefX = tile3[0];
                plDefY = tile3[1];
                mAddItem(new fFood(plDefX, plDefY, true, "w"));
                mAddItem(new fFood(plDefX, plDefY, true, "w"));
                mAddItem(new fFood(plDefX, plDefY, true, "w"));
                mAddItem(new fFood(plDefX, plDefY, true, "w"));
                mAddItem(new fFood(plDefX, plDefY, true, "w"));
                plDefX = baWAVE1_DEFENDER_SPAWN_X;
                plDefY = baWAVE1_DEFENDER_SPAWN_Y;
            }
        }

        if (baRunnersKilled < 6) {
            console.log(tile1 + ', ' + tile2 + ', ' + tile3 + ' failed with ' + baRunnersKilled + ' runners killed on pattern ' + movements);
            return false;
        }
    }

    console.log(tile1 + ', ' + tile2 + ', ' + tile3 + ' succeeded');
    return true;
}

function solveForTiles(numTiles, multiTick) {
    return;
}

function getWalkDistance(x1, y1, x2, y2) {
    return Math.max(Math.abs(x1 - x2), Math.abs(y1 - y2));
}

function isInRange(x1, y1, x2, y2, range) {
    return (getWalkDistance(x1, y1, x2, y2) <= range);
}

function isSameTile(x1, y1, x2, y2) {
    return (x1 === x2 && y1 === y2);
}

function isValidTile(tile) {

    var x = tile[0];
    var y = tile[1];

    if ((x === 47) && (y === 31)) {
        return true;
    }
    if ((x === 47) && (y === 32)) {
        return true;
    }

    if ((x === 46) && ((y >= 28) && (y <= 33))) {
        return true;
    }

    if (((x >= 43) && (x <= 46)) && ((y >= 25) && (y <= 27))) {
        return true;
    }

    if ((x === 47) && (y === 25)) {
        return true;
    }
    if ((x === 47) && (y === 26)) {
        return true;
    }
    if ((x === 48) && (y === 25)) {
        return true;
    }

    if (((x >= 34) && (x <= 45)) && ((y >= 28) && (y <= 36))) {
        return true;
    }

    if (((x >= 34) && (x <= 43)) && ((y >= 37) && (y <= 38))) {
        return true;
    }

    if ((x === 44) && (y === 37)) {
        return true;
    }

    return false;
}


//var threeTileSolution = solveForThreeTiles();
//console.log(threeTileSolution.length);
//console.log(threeTileSolution);

//var fourTileSolution = solveForFourTiles();
//console.log(fourTileSolution.length);
//console.log(fourTileSolution);

var fiveTileSolution = solveForFiveTiles();
console.log(fiveTileSolution.length);
console.log(fiveTileSolution, {'maxArrayLength': null});