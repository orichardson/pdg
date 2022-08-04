const initw = 50, inith = 40;
// const STRETCH_FACTOR = 1
const STRETCH_FACTOR = 1.2;
// const STRETCH_FACTOR = 3;

// const OPT_DIST = {0 : 35, 1:50, 2:70, 3:100, 4: 150, 5: 180, 6: 180};
// const OPT_DIST = {1:50, 2:70, 3:100, 4: 110, 5: 120, 6: 130};
// const OPT_DIST = { 1:25,  2:35,  3:50, 4: 65, 5:80, 6:95 };
const OPT_DIST = { 1:25,  2:35,  3:50, 4: 65, 5:80, 6:95 };

function default_separation(nsibls, isLoop) {
	return (nsibls in OPT_DIST ? OPT_DIST[nsibls] : 20*nsibls) + sgn(isLoop)*50;
}


function linkobject([label, [src,tgt]], i) {
	// return { "source" : src.join(","), "target" : tgt.join(","), "index": i};
	return {
		// source + target useful for using as actual force links
		source: src.join(","), 
		target: tgt.join(","), 
		index: i, 
		label: label,
		srcs : src,
		tgts : tgt,
		display: true,
		//## Added Later:
		// path2d, lw, arclen
		//## Actual Data
		cpd : null,
	}
}

function PDGView(hypergraph) {
		//data from pdgviz.js
	nodes = [];
	links = [];
	linknodes = [];
	lookup = [];
	parentLinks = [];
	
	
	simulation = undefined
	// 
	svgg = d3.select("#svg").append("g")
		.classed("PDG", true)
	
	if(hypergraph !== undefined) {
		load(hypergraph)
	}
	
	function load(hypergraph) {
		if(typeof simulation != "undefined") {
			simulation.stop();
		}
		
		// clear state
		parentLinks = [];
		lookup = [];
		nodes = [];
		linknodes = [];
		links = []
		svgg.selectAll(".node").data([]).exit().remove();
		// align_node_dom();
		
		// load nodes
		nodes = hypergraph.nodes.map( function(varname) {
			let ob = {id: varname, values: [0,1],
				w : initw, h: inith, display: true};
			lookup[varname] = ob;
			return ob;
		});
		
		// load saved node properties
		if(hypergraph.viz && hypergraph.viz.nodes) {
			for( const [nid, propobj] of Object.entries(hypergraph.viz.nodes)){
				Object.assign(lookup[nid], propobj)
			}
		}
		align_node_dom();		

		
		// load hyper-edges
		let ED = hypergraph.hedges;
		for (label in ED) {
			for(var multi of ED[label]) {
				ensure_multinode(multi);
			}
		}
		links = Object.entries(ED).map(linkobject);
		linknodes = links.map(mk_linknode)
		
		// load saved viz properties for link-nodes (e.g., link positions)
		if(hypergraph.viz && hypergraph.viz.linknodes) {
			hypergraph.viz.linknodes.forEach( function([label, ob]) {
				let ln = linknodes.find(ln => ln.link.label == label);
				Object.assign(ln, ob);
			});
		} 
		if(hypergraph.viz && hypergraph.viz.links) {
			hypergraph.viz.links.forEach( function([label, ob]) {
				let l = links.find(l => l.label == label);
				Object.assign(l, ob);
			});
		} 
		// 
		
		// if simulation exists, update nodes & edges of simulation + restart.
		if(typeof simulation != "undefined") {
			simulation.nodes(nodes.concat(linknodes));
			restyle_nodes();
			restyle_links();

			simulation.force('bipartite').links(mk_bipartite_links(links));
			
			if( ! hypergraph.viz ){
				reinitialize_node_positions();
			}
			else {
				ontick();
				simulation.alpha(0.05).restart();
			}
			
		}
	}
	
	function current_hypergraph() {
		let hedges = {}
		for(let l of links) {
			hedges[l.label] = [l.srcs, l.tgts];
		}
		return {
			nodes : nodes.map(n => n.id),
			hedges : hedges,
			viz : {
				nodes : Object.fromEntries(nodes.map(
						n => [n.id, cloneAndPluck(n, ["x", "y", "w", "h", "selected", "expanded"])]
						// n => [n.id, n]
					)),
				linknodes : linknodes.map(
					// ln =>  [ln.link.label, cloneAndPluck(ln, ["x", "y", "w", "h"] )]
					ln =>  [ln.link.label, cloneAndPluck(ln, ["x", "y", "w", "h", "sep" ] )]
				),
				links : links.map(
					l =>  [l.label, cloneAndPluck(l, ["selected" ] )]
				)
			}
		}
	}

	function ensure_multinode(multi) {
		s = multi.join(',')
		if( ! nodes.find(n => n.id == s )) {
			let ob = {id:s, 
				// w:6, h:6, display: false,
				w:2, h:2, display: false,
				components: multi,
			 	vx:0.0, vy:0.0};
			
			if( multi.length > 0 ) 
				[ob.x, ob.y] = avgpos(...multi); // defined below.
		
			// nodes.push(ob);
			lookup[s] = ob;
			multi.forEach(n =>
				parentLinks.push({"source" : s, "target" : n}) );
		};
	}
	function avgpos( ... nodenames ) {
		return   [ d3.mean(nodenames.map(v => lookup[v].x)),
					 d3.mean(nodenames.map(v => lookup[v].y)) ];
	}
	function mk_linknode(link) {
		let avg = avgpos(...link.srcs, ...link.tgts)
		let ob = {
			// id: link.label+link.source+link.target
			id: "ℓ"+link.label, 
			link: link,
			x: avg[0] + 10*Math.random()-5,
			y: avg[1] + 10*Math.random()-5,
			offset: [0,0], vx: 0, vy:0,
			// w : 5, h : 5,
			w : 0, h : 0,
			display: false};
		return ob;
	}
	
	function compute_link_shape(src, tgt, midpt=undefined,
		 	return_mid=false, arrwidth=undefined) {
		if(arrwidth == undefined) arrwidth=10;
		// let srcnode = lookup[src.join(",")];
		// let avgsrc = vec2(srcnode);
		// if( src.length > 0 ) {
		// 	avgsrc = avgpos(...src);
		// }
		let avgsrc = src.length==0 ? (midpt ? midpt : vec2(lookup[''])) : avgpos(...src);
		
		// let tgtnode = lookup[tgt.join(",")];
		// let avgtgt = vec2(tgtnode);
		// if( tgt.length == 0 ) {
		// 	avgtgt = avgpos(...tgt);
		// }
		let avgtgt = tgt.length==0 ? (midpt ? midpt : vec2(lookup[''])) : avgpos(...tgt);

		// let mid = [ 0.4*avgsrc[0] + 0.6*avgtgt[0], 0.4*avgsrc[1] + 0.6*avgtgt[1] ];
		// let mid = midpt ? midpt : 
		let mid = [ 0.4*avgsrc[0] + 0.6*avgtgt[0], 0.4*avgsrc[1] + 0.6*avgtgt[1] ];
		// console.log('ho', avgsrc,avgtgt, mid);
		function shortener(s) {
			return sqshortened_end(mid, vec2(lookup[s]), [lookup[s].w, lookup[s].h], 10);
		}
		let avgsrcshortened = src.length == 0 ? 
			(midpt ? midpt: shortener("")) : scale(addv(... src.map(shortener)), 1 / src.length);
		let avgtgtshortened = tgt.length == 0 ?
			(midpt ? midpt: shortener("")) : scale(addv(... tgt.map(shortener)), 1 / tgt.length);
		let midearly = mid;
		// mid = [ .5*avgsrcshortened[0] + .5*avgtgtshortened[0],
		// 	.5*avgsrcshortened[1] + .5*avgtgtshortened[1] ];
		let true_mid = [ .5*avgsrcshortened[0] + .5*avgtgtshortened[0],
			.5*avgsrcshortened[1] + .5*avgtgtshortened[1] ];
		mid = midpt ? midpt : true_mid;
		// mid = true_mid;
		let delta = subv(mid, true_mid);
		// let avgtgtshortened = addv(
		// 	...tgt.map(t =>
		// 		sqshortened_end(mid, vec2(lookup[t]), [lookup[t].w, lookup[t].h]))
		// 			/ tgt.length
		// );
		
		let lpath = new Path2D();
		src.forEach( function(s) {
			// lpath.moveTo(...shortener(s));
			// lpath.moveTo(lookup[s].x, lookup[s].y);
			startpt = shortener(s);
			lpath.moveTo(...startpt);
			// lpath.quadraticCurveTo(avgsrcshortened[0], avgsrcshortened[1], mid[0], mid[1]);
			// lpath.bezierCurveTo(
			// 		// avgtgt[0], avgtgt[1],
			// 		0.2*midearly[0] + startpt[0]*(0.8),
			// 		0.2*midearly[1] + startpt[1]*(0.8),
			// 		.8*avgsrcshortened[0] + mid[0]*(0.2),
			// 		.8*avgsrcshortened[1] + mid[1]*(0.2),
			// 		// lookup[s].x, lookup[s].y,
			// 		mid[0], mid[1]);
			// lpath.bezierCurveTo(
			// 		// avgtgt[0], avgtgt[1],
			// 		0.2*midearly[0] + startpt[0]*(0.8) + delta[0] * 0.9,
			// 		0.2*midearly[1] + startpt[1]*(0.8) + delta[1] * 0.9,
			// 		.8*avgsrcshortened[0] + true_mid[0]*(0.2) + delta[0] * 1.8,
			// 		.8*avgsrcshortened[1] + true_mid[1]*(0.2) + delta[1] * 1.8,
			// 		// lookup[s].x, lookup[s].y,
			// 		mid[0], mid[1]);
			
			
			lpath.bezierCurveTo(
					// avgtgt[0], avgtgt[1],
					// 0.2*midearly[0] + startpt[0]*(0.8) + delta[0] * 0,
					// 0.2*midearly[1] + startpt[1]*(0.8) + delta[1] * 0,
					 0.2*mid[0] + startpt[0]*(0.8) + delta[0] * 0,
					 0.2*mid[1] + startpt[1]*(0.8) + delta[1] * 0,
					.8*avgsrcshortened[0] + true_mid[0]*(0.2) + delta[0],
					.8*avgsrcshortened[1] + true_mid[1]*(0.2) + delta[1],
					// lookup[s].x, lookup[s].y,
					mid[0], mid[1]);

			// lpath.moveTo(...startpt);
			// lpath.lineTo(mid[0], mid[1]);
		});
		tgt.forEach( function(t) {
			// lpath.moveTo( true_mid[0], true_mid[1] );
			lpath.moveTo(...mid);
			// lpath.quadraticCurveTo(avgtgt[0], avgtgt[1], lookup[t].x, lookup[t].y);
			let endpt = shortener(t);
			// console.log(mid, vec2(lookup[t]), endpt);
			// scale(delta, Math.max(0, norm-35) / norm )

			// lpath.quadraticCurveTo(avgtgtshortened[0], avgtgtshortened[1], endpt[0], endpt[1]);
			// lpath.lineTo(...endpt);
			central_ctrl = [
					.8*avgtgtshortened[0] + true_mid[0]*(0.2) + delta[0],
					.8*avgtgtshortened[1] + true_mid[1]*(0.2) + delta[1],
				];
			proximal_ctrl =  [
					0.2*mid[0] + endpt[0]*(0.8) + delta[0] * 0,
					0.2*mid[1] + endpt[1]*(0.8) + delta[1] * 0 
				];
			lpath.bezierCurveTo(...central_ctrl, ...proximal_ctrl, ...endpt);
					// lookup[s].x, lookup[s].y,
					// endpt[0], endpt[1]);
			
			
			
			// ### DRAW ARROWS
			// let [ar0, ar1, armid0, armid1] = arrowpts(mid, endpt, arrwidth);
			// let [ar0, ar1, armid0, armid1] = arrowpts(central_ctrl, endpt, arrwidth);
			let [ar0, ar1, armid0, armid1] = arrowpts(meldv(central_ctrl,proximal_ctrl,0.8), endpt, arrwidth);
			lpath.moveTo(...endpt);
			lpath.quadraticCurveTo(armid0[0], armid0[1], ar0[0], ar0[1]);
			lpath.moveTo(...endpt);
			lpath.quadraticCurveTo(armid1[0], armid1[1], ar1[0], ar1[1]);
		});
		if(return_mid) return [lpath, true_mid];
		return lpath;
	}

	function ontick() {
		// for (let l of links) {
		// 	l.path2d = compute_link_shape(l.srcs,l.tgts);
		// }
		for (let ln of linknodes) {
			let l = ln.link;
			[l.path2d, ln.true_mid] = compute_link_shape(l.srcs, l.tgts, vec2(ln), true, (l.lw|2)*1.5+6);
		}

		// clamp to within boundary
		nodes.concat(linknodes).forEach(function(n) {
			n.x = clamp(n.x, n.w/2, canvas.width - n.w/2);
			n.y = clamp(n.y, n.h/2, canvas.height - n.h/2);
		});
		
		restyle_nodes();
		restyle_links();
		redraw();
	}

	
	return {
		sim : simulation,
		load : load,
		tick : ontick
	}
	
}
