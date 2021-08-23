console.log("Custom JS Executing");

////  here's the hypergraph...
// let [N, ED] = [["PS", "S", "SH", "C"], [[[], ["PS"]], [["PS"], ["S"]], [["PS"], ["SH"]], [["S", "SH"], ["C"]]]];
////version with "1" instead of empy set.
// let [N, ED] = [["1", "PS", "S", "SH", "C"], [[["1"], ["PS"]], [["PS"], ["S"]], [["PS"], ["SH"]], [["S", "SH"], ["C"]]]];
//// version with extra node T
// let [N, ED] = [["PS", "S", "SH", "C", "T"], [[[], ["PS"]], [["PS"], ["S"]], [["PS"], ["SH"]], [["S", "SH"], ["C"]]]];
//// test new format
var hypergraph = {
	nodes : ["PS", "S", "SH", "C", "T", "Test 1", "Test 2"], 
	hedges : {0: [[], ["PS"]], 
	 1: [["PS"], ["S"]],
	 2: [["PS"], ["SH"]],
	 3: [["S", "SH"], ["C"]], 
	 P: [["T"], ["Test 1", "Test 2"]] } 
};
// hypergraph = {
// 	nodes : ['A', 'B', 'C', 'D'],
// 	hedges : {
// 		p0: [['B', 'C'], ['A']],
// 		p2: [['A', 'D'], ['B']],
// 		p4: [['A', 'D'], ['C']],
// 		p6: [['B', 'C'], ['D']]
// 	}
// };


let [N, ED] = [hypergraph.nodes, hypergraph.hedges];

		

const initw = 60, inith = 40;

$(function() {
	// resize to full screen
	let canvas = document.getElementById("canvas"),
		svg = d3.select("#svg");

	function resizeCanvas() {
		canvas.width = window.innerWidth;
		canvas.height = window.innerHeight;
	}
	resizeCanvas()
	window.addEventListener('resize', resizeCanvas, false);


	context = canvas.getContext("2d");
	lookup = []
	nodes = N.map( function(varname) {
		let ob = {"id": varname, "w" : initw, "h": inith, "display": true};
		lookup[varname] = ob;
		return ob;
	});
	fullN = [...N];
	parentLinks = [];
	
	for (label in ED) {
		for(var multi of ED[label]) {
			s = multi.join(',')
			if( !fullN.includes(s)) {
				let ob = {"id": s, "w" : 6, "h": 6, "display": false};
				nodes.push(ob);
				fullN.push(s);
				lookup[s] = ob;
				multi.forEach(n =>
					parentLinks.push({"source" : s, "target" : n}) );
			}
		}
	}
	// window.lookup = lookup
	// hlinks = ED.map(


	nodedata = svg.selectAll(".node").data(nodes);
	gnode = nodedata.enter().append("g").classed("node", true);
	gnode.append("rect")
		.classed("nodeshape", true)
		.attr('width', n => n.w).attr('x', n => -n.w/2)
		.attr('height', n => n.h).attr('y', n => -n.h/2)
		.attr('rx', 15);
	gnode.append("text").text(n => n.id);
	gnode.filter( n => ! n.display).attr('display', 'none')

	//##  Next, draw PDG.
	function ontick() {
		context.clearRect(0, 0, canvas.width, canvas.height);


		
		for (label in ED) {
			let [src, tgt] = ED[label]
			
			// console.log(src, tgt, lookup);
			srcnode = lookup[src.join(",")];
			let avgsrc = vec2(srcnode);
			if( src.length > 0 ) {
				avgsrc = [ d3.mean(src.map(v => lookup[v].x)),
										d3.mean(src.map(v => lookup[v].y)) ];
				srcnode.vx += (avgsrc[0] - srcnode.x) * 0.2
				srcnode.vy += (avgsrc[1] - srcnode.y) * 0.2
			}
			tgtnode = lookup[tgt.join(",")];
			let avgtgt = vec2(tgtnode);
			// if( tgt.length > 0 ) {
			// 	srcnode.vx += (avgsrc[0] - srcnode.x) * 0.2
			// 	srcnode.vy += (avgsrc[1] - srcnode.y) * 0.2
			// 	avgtgt = [ d3.mean(tgt.map(v => lookup[v].x)),
			// 							d3.mean(tgt.map(v => lookup[v].y)) ];
			// }
			if( tgt.length > 0 ) {
				avgtgt = [ d3.mean(tgt.map(v => lookup[v].x)),
										d3.mean(tgt.map(v => lookup[v].y)) ];
				tgtnode.vx += (avgtgt[0] - tgtnode.x) * 0.2
				tgtnode.vy += (avgtgt[1] - tgtnode.y) * 0.2
			}

			let mid = [ 0.4*avgsrc[0] + 0.6*avgtgt[0], 0.4*avgsrc[1] + 0.6*avgtgt[1] ];
			// console.log('ho', avgsrc,avgtgt, mid);
			const shortener = s =>
				sqshortened_end(mid, vec2(lookup[s]), [lookup[s].w, lookup[s].h], 10);
			let avgsrcshortened = src.length == 0 ? 
				shortener("") : scale(addv(... src.map(shortener)), 1 / src.length);
			let avgtgtshortened = tgt.length == 0 ?
				shortener("") : scale(addv(... tgt.map(shortener)), 1 / tgt.length);
			let midearly = mid;
			mid = [ .5*avgsrcshortened[0] + .5*avgtgtshortened[0],
				.5*avgsrcshortened[1] + .5*avgtgtshortened[1] ];
			// let avgtgtshortened = addv(
			// 	...tgt.map(t =>
			// 		sqshortened_end(mid, vec2(lookup[t]), [lookup[t].w, lookup[t].h]))
			// 			/ tgt.length
			// );
			
			// if(!srcnode.id){
			// 	console.log(midearly, mid, avgsrc, avgtgt);
			// }
			// console.log(srcnode.id, srcnode.x, srcnode.y, '\t', tgtnode.id,  tgtnode.x, tgtnode.y);
			// console.log(avgtgtshortened);
			
			// context.lineWidth = 1;
			// context.setLineDash([4,1]);
			// context.strokeStyle = 'red';
			// context.beginPath();
			// // context.moveTo(srcnode.x, srcnode.y);
			// // context.lineTo(tgtnode.x, tgtnode.y);
			// context.moveTo(...avgsrcshortened);
			// context.lineTo(...avgtgtshortened);
			// context.stroke();
			

			context.lineWidth = 1.5;
			context.strokeStyle = "black";
			context.setLineDash([]);
			context.beginPath();
			src.forEach( function(s) {
				// context.moveTo(...shortener(s));
				// context.moveTo(lookup[s].x, lookup[s].y);
				startpt = shortener(s);
				context.moveTo(...startpt);
				// context.quadraticCurveTo(avgsrcshortened[0], avgsrcshortened[1], mid[0], mid[1]);
				context.bezierCurveTo(
						// avgtgt[0], avgtgt[1],
						0.2*midearly[0] + startpt[0]*(0.8),
						0.2*midearly[1] + startpt[1]*(0.8),
						.8*avgsrcshortened[0] + mid[0]*(0.2),
						.8*avgsrcshortened[1] + mid[1]*(0.2),
						// lookup[s].x, lookup[s].y,
						mid[0], mid[1]);

				// context.lineTo(mid[0], mid[1]);
			});
			tgt.forEach( function(t) {
				context.moveTo( mid[0], mid[1] );
				// context.quadraticCurveTo(avgtgt[0], avgtgt[1], lookup[t].x, lookup[t].y);
				let endpt = shortener(t);
				// console.log(mid, vec2(lookup[t]), endpt);
				// scale(delta, Math.max(0, norm-35) / norm )

				// context.quadraticCurveTo(avgtgtshortened[0], avgtgtshortened[1], endpt[0], endpt[1]);
				context.lineTo(...endpt);
				let [ar0, ar1, armid0, armid1] = arrowpts(mid, endpt, 10);
				context.moveTo(...endpt);
				context.quadraticCurveTo(armid0[0], armid0[1], ar0[0], ar0[1]);
				context.moveTo(...endpt);
				context.quadraticCurveTo(armid1[0], armid1[1], ar1[0], ar1[1]);
			});
			context.stroke();
		}

		// context.stroke();
		// context.fill();
		// context.restore();
		// nodes.forEach(function(n) {
		// 	context.moveTo(n.x, n.y);
		// 	context.arc(n.x, n.y, 3, 0, 2 * Math.PI);
		// });
		
		context.globalAlpha = 0.2;
		fullN.forEach(function(nn) {
			n = lookup[nn];			
			//updating
			n.x = clamp(n.x, n.w/2, canvas.width - n.w/2);
			n.y = clamp(n.y, n.h/2, canvas.height - n.h/2);
			//drawing
			if(! N.includes(nn)) {
				context.moveTo(n.x, n.y);
				context.arc(n.x, n.y, 3, 0, 2 * Math.PI);
				context.stroke();
			}
		});
		context.globalAlpha = 1;

		context.save();
		// console.log(canvas.width, canvas.height);


		/*** Now for some SVG operations. ***/

		// svg.selectAll(".node").data(nodes)#.call(
		svg.selectAll(".node").data(N)
			.attr("transform", n => "translate(" + lookup[n].x + ","+lookup[n].y +")");
		 	// .attr("cx", n => n.x)
			// .attr("cy", n => n.y);
			// .select("circle")



	}


	links = Object.entries(ED).map(function([label,[src,tgt]],i) {
		// return { "source" : src.join(","), "target" : tgt.join(","), "index": i};
		return { source: src.join(","), target: tgt.join(","), index: i, label: label};
	});
	
	function customForces() {
		
	}
	

	simulation = d3.forceSimulation(nodes)
		.force("center",
			d3.forceCenter(canvas.width / 2, canvas.height / 2).strength(0.03))
		.force("charge", d3.forceManyBody().strength(
			n => n.display ? -100 : -100))
		.force("link", d3.forceLink(links).id(l => l.id)
			.strength(1).distance(110).iterations(3))
		.force("anotherlink", d3.forceLink(parentLinks).id(n=>n.id)
				.strength(0.3).distance(40).iterations(2))
		.force("nointersect", d3.forceCollide().radius(n => n.display ? n.w/2 : 0)
				.strength(0.5).iterations(5))
		.on("tick", ontick)
		.stop();
	simulation.alphaDecay(0.01);
		
	setTimeout(function(){
		for (const node of nodes) {
			node.x = node.x * 10.8 + canvas.width/2;
			node.y = node.y * 10.8 + canvas.height/2;
		} ontick();} , 1);
	setTimeout(() => {simulation.restart(); }, 100);


	d3.select(canvas)
	    .call(d3.drag()
	        .container(canvas)
	        .subject(function(event) {
						// for(let n of N) {
							// let loon = lookup[n];
					for(let objn of nodes) {
						adx = Math.abs(objn.x - event.x);
						ady = Math.abs(objn.y - event.y);

						if(adx <  objn.w/2 && ady < objn.h/2)
							return objn;
					}
				})
	        .on("start", dragstarted)
	        .on("drag", dragged)
	        .on("end", dragended)
		);
		
	canvas.addEventListener('click', function(event) {
		
	});


	function dragstarted(event) {
	  if (!event.active) simulation.alphaTarget(0.5).restart();
	  event.subject.fx = event.subject.x;
	  event.subject.fy = event.subject.y;
	}

	function dragged(event) {
	  event.subject.fx = event.x;
	  event.subject.fy = event.y;
	}

	function dragended(event) {
	  if (!event.active) simulation.alphaTarget(0);
	  event.subject.fx = null;
	  event.subject.fy = null;
	}
});



function vec2( obj ) {
	return [obj.x, obj.y];
}
function mag( deltas ) {
	return Math.sqrt(d3.sum( deltas.map(dx => dx*dx) ));
}
function subv(v1, v2) {
	// return [v1[0]-v2[0], v1[1]-v2[1]];
	return v1.map( (v1i, i) => v1i - v2[i]);
}
function addv(...vecs) {
	// return v1.map( (v1i, i) => v1i + v2[i]);
	// return [x1+x2, y1+y2];
	if(vecs.length > 0) {
		// console.log(vecs[0]);
		return vecs[0].map( (_v0i, i) => vecs.reduce((total, v) => total + v[i], 0) );
	}
	return [0];
}
function scale(x, s) {
	return x.map( x => x*s);
}
function clamp(val, l, u) {
	if(val < l) return l;
	if(val > u) return u;
	return val
}
function magclamp(val, m) {
	return clamp(val, -m, m)
}
function sgn(x) {
	if(x > 0) return 1;
	if(x < 0) return -1;
	return 0;
}

function sqshortened_end(from, to, [w,h], extra=0) {
	delta = subv(to,from);
	h = h + extra;
	w = w + extra;
	sqshortdelta = [
		delta[0] - magclamp(delta[0] * h / Math.abs(delta[1]), w)/2,
		delta[1] - magclamp(delta[1] * w / Math.abs(delta[0]), h)/2];
	return addv(from,sqshortdelta);
}

function arrowpts(from, to, arrscale=20, narrow=0.75) {
	delta = subv(to,from);
	factor = arrscale / mag(delta);
	npara = scale(delta, -factor);
	base = addv(to, npara);

	ortho = [delta[1] * factor*narrow, -delta[0] * factor*narrow];
	halfortho = scale(ortho,0.5);
	return [ addv(base, ortho), subv(base, ortho),
			addv(base,halfortho), subv(base,halfortho) ];
}
