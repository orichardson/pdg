console.log("Custom JS Loaded");

//  here's the hypergraph...
let [N, ED] = [["PS", "S", "SH", "C"], [[[], ["PS"]], [["PS"], ["S"]], [["PS"], ["SH"]], [["S", "SH"], ["C"]]]];
// let [N, ED] = [["1", "PS", "S", "SH", "C"], [[["1"], ["PS"]], [["PS"], ["S"]], [["PS"], ["SH"]], [["S", "SH"], ["C"]]]];
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
	ED.forEach(function(srctgt){
		for(var multi of srctgt) {
			s = multi.join(',')
			if( !fullN.includes(s)) {
				let ob = {"id": s, "w" : 0, "h": 0, "display": false};
				nodes.push(ob);
				fullN.push(s);
				lookup[s] = ob;
				multi.forEach(n =>
					parentLinks.push({"source" : s, "target" : n}) );
			}
		}
	});
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

	//##  Next, draw PDG.
	function ontick() {
		context.clearRect(0, 0, canvas.width, canvas.height);
		context.save();


		ED.forEach(function([src,tgt]) {
			context.lineWidth = 1.5;
			context.beginPath();
			// console.log(src, tgt, lookup);
			let avgsrc = [ d3.mean(src.map(v => lookup[v].x)),
									d3.mean(src.map(v => lookup[v].y)) ];
			let avgtgt = [ d3.mean(tgt.map(v => lookup[v].x)),
									d3.mean(tgt.map(v => lookup[v].y)) ];
			srcnode = lookup[src.join(",")];
			srcnode.fx = avgsrc[0]*1;
			srcnode.fy = avgsrc[1]*1;

			let mid = [ 0.4*avgsrc[0] + 0.6*avgtgt[0], 0.4*avgsrc[1] + 0.6*avgtgt[1] ];
			// console.log('ho', avgsrc,avgtgt, mid);
			const shortener = s =>
				sqshortened_end(mid, vec2(lookup[s]), [lookup[s].w, lookup[s].h], 10);
			let avgsrcshortened = scale(addv(... src.map(shortener)), 1 / src.length);
			let avgtgtshortened = scale(addv(... tgt.map(shortener)), 1 / tgt.length);
			let midearly = mid;
			mid = [ .5*avgsrcshortened[0] + .5*avgtgtshortened[0],
				.5*avgsrcshortened[1] + .5*avgtgtshortened[1] ];
			// let avgtgtshortened = addv(
			// 	...tgt.map(t =>
			// 		sqshortened_end(mid, vec2(lookup[t]), [lookup[t].w, lookup[t].h]))
			// 			/ tgt.length
			// );

			src.forEach( function(s) {
				// context.moveTo(...shortener(s));
				// context.moveTo(lookup[s].x, lookup[s].y);
				startpt = shortener(s);
				context.moveTo(...startpt);
				context.quadraticCurveTo(avgsrcshortened[0], avgsrcshortened[1], mid[0], mid[1]);
				// context.bezierCurveTo(
				// 		// avgtgt[0], avgtgt[1],
				// 		midearly[0]*0.4 + startpt[0]*0.6,midearly[1]*0.4+startpt[1]*0.6,
				// 		avgsrcshortened[0]*1.2+mid[0]*(-0.2),
				// 			1.2*avgsrcshortened[1]+mid[1]*(-0.2),
				// 		// lookup[s].x, lookup[s].y,
				// 		mid[0], mid[1]);

				// context.lineTo(mid[0], mid[1]);
			});
			tgt.forEach( function(t) {
				context.moveTo( mid[0], mid[1] );
				// context.quadraticCurveTo(avgtgt[0], avgtgt[1], lookup[t].x, lookup[t].y);
				let endpt = shortener(t);
				// console.log(mid, vec2(lookup[t]), endpt);
				// scale(delta, Math.max(0, norm-35) / norm )

				context.quadraticCurveTo(avgtgtshortened[0], avgtgtshortened[1], endpt[0], endpt[1]);
				// context.lineTo(...endpt);
				let [ar0, ar1, armid0, armid1] = arrowpts(mid, endpt, 10);
				context.moveTo(...endpt);
				context.quadraticCurveTo(armid0[0], armid0[1], ar0[0], ar0[1]);
				context.moveTo(...endpt);
				context.quadraticCurveTo(armid1[0], armid1[1], ar1[0], ar1[1]);
			});
			context.stroke();
		});

		// context.stroke();
		// context.fill();
		// context.restore();
		// nodes.forEach(function(n) {
		// 	context.moveTo(n.x, n.y);
		// 	context.arc(n.x, n.y, 3, 0, 2 * Math.PI);
		// });
		fullN.forEach(function(nn) {
			n = lookup[nn];
			context.moveTo(n.x, n.y);
			context.arc(n.x, n.y, 3, 0, 2 * Math.PI);
			context.stroke();
		});



		// svg.selectAll(".node").data(nodes)#.call(
		svg.selectAll(".node").data(N)
			.attr("transform", n => "translate(" + lookup[n].x + ","+lookup[n].y +")");
		 	// .attr("cx", n => n.x)
			// .attr("cy", n => n.y);
			// .select("circle")



	}


	links = ED.map(function([src,tgt],i) {
		console.log(i);
		return { "source" : src.join(","), "target" : tgt.join(","), "index": i};
	});

	simulation = d3.forceSimulation(nodes)
		// .force("center",
		// 	d3.forceCenter(canvas.width / 2, canvas.height / 2))
		.force("charge", d3.forceManyBody().strength(-50))
		.force("link", d3.forceLink(links).id(n=>n.id)
			.strength(1).distance(110).iterations(2))
		.force("anotherlink", d3.forceLink(parentLinks).id(n=>n.id)
				.strength(0.2).distance(50).iterations(2))
		.force("nointersect", d3.forceCollide().radius(n=>n.w/2)
				.strength(0.5).iterations(5));
	simulation.on("tick", ontick);

		d3.select(canvas)
		    .call(d3.drag()
		        .container(canvas)
		        .subject(function() {
							for(let n of N) {
								let loon = lookup[n];
								adx = Math.abs(loon.x - d3.event.x);
								ady = Math.abs(loon.y - d3.event.y);

								if(adx <  loon.w/2 && ady < loon.h/2)
									return loon;
							}
						})
		        .on("start", dragstarted)
		        .on("drag", dragged)
		        .on("end", dragended)
					);


		function dragstarted() {
		  if (!d3.event.active) simulation.alphaTarget(0.5).restart();
		  d3.event.subject.fx = d3.event.subject.x;
		  d3.event.subject.fy = d3.event.subject.y;
		}

		function dragged() {
		  d3.event.subject.fx = d3.event.x;
		  d3.event.subject.fy = d3.event.y;
		}

		function dragended() {
		  if (!d3.event.active) simulation.alphaTarget(0);
		  d3.event.subject.fx = null;
		  d3.event.subject.fy = null;
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
		return vecs[0].map( (v0i, i) => vecs.reduce((total, v) => total + v[i], 0) );
	}
	return [0];
}
function scale(x, s) {
	return x.map( x => x*s);
}
function magclamp(val, m) {
	if(val > m) return m;
	if(val < -m) return -m;
	return val;
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
