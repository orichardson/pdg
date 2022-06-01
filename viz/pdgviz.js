//A web-page opened via the file:// protocol cannot use import / export.
// import defaultExport from '/link-modified.js';

console.log("pdgvz.js");
hypergraph = {
	nodes : ['A', 'B', 'C', 'D'],
	hedges : {
		p0: [['B', 'C'], ['A']],
		$p_2$: [['A', 'D'], ['B']],
		p4: [['A', 'D'], ['C']],
		p6: [['B', 'C'], ['D']]
	}
};


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

$(function() {
	// resize to full screen
	let canvas = document.getElementById("canvas"),
		svg = d3.select("#svg");
	let context = canvas.getContext("2d");

	function resizeCanvas() {
		canvas.width = window.innerWidth;
		canvas.height = window.innerHeight;
		if(typeof simulation != "undefined") {
			// simulation.force('center').x(canvas.width/2);
			// simulation.force('center').y(canvas.height/2);
			simulation.alpha(1).restart();
			ontick();
		}
		// ontick();
	}
	window.addEventListener('resize', resizeCanvas, false);
	resizeCanvas()
	
	let mode = $('#drag-mode-toolbar button.active').attr('data-mode');
	
	$('#drag-mode-toolbar button').on('click', function() {
		$('#drag-mode-toolbar button').removeClass("active");
		$(this).addClass('active');
		mode = $(this).attr('data-mode');
		// console.log('new mode: ', mode);
	});

	// TODO LATER: make these lets.
	nodes = [];
	links = [];
	lookup = [];
	linknodes = [];
	let parentLinks = [];
	
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
	
	load_hypergraph(hypergraph);
	
	function load_hypergraph(hypergraph) {
		if(typeof simulation != "undefined") {
			simulation.stop();
		}
		
		// clear state
		parentLinks = [];
		lookup = [];
		nodes = [];
		linknodes = [];
		links = []
		svg.selectAll(".node").data([]).exit().remove();
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
	
	$('#save-button').click(function(e){
		download_JSON(current_hypergraph(), 'hypergraph');
	});
	$('#load-button').click(function(e){
		$('#fileupload').click();
	})
	$('#fileupload').on('change', function(evt){
		// console.log(evt);
		const reader = new FileReader();
		reader.onload = function(e) {
			// console.log(e);
			let ob = JSON.parse(e.target.result);
			load_hypergraph(ob);
			// console.log("LOADED HYPERGRAPH:", ob);
		};
		reader.readAsText(evt.target.files[0]);
	})	
	
	// temporary states, for actions
	var temp_link = null;	
	var popup_process = null;
	var popped_up_link = null;
	var action = {};
 
	//##  Next, Updating + Preparing shapes for drawing, starting with a 
	// helpful way of getting average position by node labels.  
	function avgpos( ... nodenames ) {
		// if ( nodenames[0] == "<MOUSE>")
		// 	return mouse_pt;
		return   [ d3.mean(nodenames.map(v => lookup[v].x)),
				   d3.mean(nodenames.map(v => lookup[v].y)) ];
	}
	function compute_link_shape(src, tgt, midpt=undefined, return_mid=false, arrwidth=undefined) {
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
	function redraw() {
		context.save();
		context.clearRect(0, 0, canvas.width, canvas.height);
		
		context.lineWidth = 1.5;
		context.strokeStyle = "black";

		context.lineCap = 'round';
		// context.setLineDash([]);

		for( let l of links) {
			// let lw = l.hasAttribute('lw')? l.lw : 2;
			if(!l.display) continue;
			let lw = l.lw | 2;
			context.lineWidth = lw * 1.2 + 3;
			context.strokeStyle = l.selected ? "rgba(230, 150, 50, 0.4)" : "rgba(255, 255, 255, 0.7)";
			context.stroke(l.path2d);
			
			context.lineWidth =  lw;
			context.strokeStyle = l.selected ? "#863" : "black";
			context.stroke(l.path2d);
			// context.lineWidth = 1;
			// context.setLineDash([4,1]);
			// context.strokeStyle = 'red';
			// context.beginPath();
			// // context.moveTo(srcnode.x, srcnode.y);
			// // context.lineTo(tgtnode.x, tgtnode.y);
			// context.moveTo(...avgsrcshortened);
			// context.lineTo(...avgtgtshortened);
			// context.stroke();
		}
		
		if(temp_link) {
			let midpt = (temp_link.x == undefined) ? undefined : vec2(temp_link);
			let tlpath = compute_link_shape(temp_link.srcs, temp_link.tgts, midpt);
			
			context.lineWidth = 3;
			context.strokeStyle = "rgba(255,255,255,0.4)";
			context.stroke( tlpath )
			
			context.lineWidth = 1.5;
			context.strokeStyle = "black";
			context.stroke( tlpath )
		}
		context.restore();
		context.save();
		// Draw Selection Rectangle
		context.globalAlpha = 0.2;
		if( action.type == 'box-select' && action.start) {
			// console.log(...corners2xywh(select_rect_start, select_rect_end))
			// context.save();
			context.fillStyle="orange";
			
			// context.fillRect(select_rect_start.x, select_rect_start.y, select_rect_end.x, select_rect_end.y);
			// let [xmin,ymin,w,h] = corners2xywh(select_rect_start, select_rect_end);
			context.fillRect(...corners2xywh(action.start, action.end));
			// context.stroke();
			// context.restore();
		}
		
		
		//DEBUG: Draw ex and ey of nodes
		// context.globalAlpha = 0.7;
		// for (let n of nodes) {
		// 	if(n.ex && n.ey) {
		// 		context.fillStyle="#A4C";
		// 		context.beginPath();
		// 		context.arc(n.ex, n.ey, 10, 0, 2 * Math.PI);
		// 		context.fill();
		// 	}
		// 	if(n.ex2 && n.ey2) {
		// 		context.fillStyle="#CA4";
		// 		context.beginPath();
		// 		context.arc(n.ex2, n.ey2, 10, 0, 2 * Math.PI);
		// 		context.fill();
		// 	}
		// }
		
		/// Draw the invisible product nodes + make sure no node goes off screen.
		context.globalAlpha = 0.5;
		context.lineWidth = 2;
		nodes.forEach(function(n) {
			if(! n.display ) {
				context.beginPath();

				if( n.selected )
					context.strokeStyle="#EA2";
				else context.strokeStyle="#AAA";
				
				context.moveTo(n.x, n.y);
				context.arc(n.x, n.y, 3, 0, 2 * Math.PI);
				context.stroke();				
			}
		});

		context.fillStyle="#888";
		
		//draw the linknodes 
		// linknodes.forEach(function(n) {
		// 	context.moveTo(n.x, n.y);
		// 	context.beginPath();
		// 	// context.fillStyle="#A4C";
		// 	context.arc(n.x, n.y, 7, 0, 2 * Math.PI);
		// 	context.fill();				
		// });
		// context.globalAlpha = 1;
		context.restore();
	}
	
	// Simlation Functions: forces, and initialization
	function multi_avgpos_alignment_force(alpha) {
		for(let n of nodes) {
			if (n.components && n.components.length > 1) {
				// console.log('working?');
				let avg = avgpos(...n.components);
				// let delta = subv(avg, vec2(n));
				// let scale = Math.pow(mag(delta), alpha);
				// n.vx += sgn(avg[0] - n.x) * scale * 1
				// n.vy += sgn(avg[1] - n.y) * scale * 1 
				
				n.vx += (avg[0] - n.x) * 0.5 * alpha;
				n.vy += (avg[1] - n.y) * 0.5 * alpha;
				
				// n.vx += (avg[0] - n.x) * 0.3;
				// n.vy += (avg[1] - n.y) * 0.3;
			}
		}
		// now even out forces
		// for( let l of links) {
			//// TODO softly even out distances between components across links.
		// }
	}
	function midpoint_aligning_force(alpha) {
		// let strength = 0.3 // 0.35
		let strength = 0.2 // 0.35
		for (let ln of linknodes) {
			let l = ln.link;
			if(l.srcs.length ==0) continue;
			[l.path2d, ln.true_mid] = compute_link_shape(l.srcs, l.tgts, vec2(ln), true);
			// ln.x += (mid[0] - ln.x) * 0.25;
			// ln.y += (mid[1] - ln.y) * 0.25;
			// ln.vx += (ln.true_mid[0] + ln.offset[0] - ln.x - ln.vx) * strength * alpha; 
			// ln.vy += (ln.true_mid[1] + ln.offset[1] - ln.y - ln.vy) * strength * alpha;
			
			// ln.x += (ln.true_mid[0] + ln.offset[0] - ln.x - ln.vx) * strength * alpha; 
			// ln.y += (ln.true_mid[1] + ln.offset[1] - ln.y - ln.vy) * strength * alpha;
			
			ln.vx += (ln.true_mid[0] + ln.offset[0] - ln.x - ln.vx) * strength * alpha; 
			ln.vy += (ln.true_mid[1] + ln.offset[1] - ln.y - ln.vy) * strength * alpha;
		
			// ln.x += (ln.true_mid[0] + ln.offset[0] - ln.x) * strength * alpha; 
			// ln.y += (ln.true_mid[1] + ln.offset[1] - ln.y) * strength * alpha;
		}
	}
	
	function mk_bipartite_links(links){
		bipartite_links = []
		// for( let l of links) {
		// let lname = "ℓ" + l.label;
		for( let ln of linknodes) {
			let l = ln.link;
			let lname = ln.id;
			
			for( let s of l.srcs) {
				bipartite_links.push({ 
					source: s, target: lname, 
					n : s, ln : ln,
					separation : 
						ln.sep && ln.sep[s] ? ln.sep[s] : 
						default_separation(l.srcs.length, l.tgts.includes(s)),
					// nsibls: l.srcs.length, 
					// isloop: l.tgts.includes(s)
				});
			}
			// delta = l.srcs.length == 0 ?  -1 : 0;
			for( let t of l.tgts) {
				bipartite_links.push({
					source: lname, target: t, 
					n : t, ln : ln,
					separation : 
						ln.sep && ln.sep[t] ? ln.sep[t] : 
						default_separation(l.tgts.length, l.srcs.includes(t)),
					// nsibls: l.tgts.length, 
					// isloop: l.srcs.includes(t) 
				});
			}
		}
		return bipartite_links;
	}
	function reinitialize_node_positions() {
		for (let node of nodes) {
			node.x = node.x * 10.8 + canvas.width/2;
			node.y = node.y * 10.8 + canvas.height/2;
		} 
		for(let ln of linknodes) {
			tgtavg = avgpos(...ln.link.tgts);
			if(ln.link.srcs.length == 0)
				[ln.x, ln.y] = tgtavg;
			else
				[ln.x, ln.y] = scale( addv(tgtavg, avgpos(...ln.link.srcs)), 0.5);
		}
		ontick(); 
		simulation.alpha(2).restart();
	} 
	
	simulation = d3.forceSimulation(nodes.concat(linknodes))
		//// .force("charge", d3.forceManyBody().strength( -100))
		// .force("link", d3.forceLink(links).id(l => l.id)
		// 	.strength(1).distance(110).iterations(3))
		// .force("anotherlink", d3.forceLink(parentLinks).id(n=>n.id)
		// 		.strength(0.3).distance(40).iterations(2))d
		// .force("avgpos_align", multi_avgpos_alignment_force)
		.force("charge", filtered_force(d3.forceManyBody()
			// .strength(n => n.display ? -100 : 0)
			// .strength(n => n.link || n.components ? 0 : -120)
			// .strength(n => (n.link || !n.display) ? 0 : -100)
			.strength(-100)
			.distanceMax(150), n => (!n.link && n.display) ))
		.force("linkcharge", filtered_force(d3.forceManyBody()
			// .strength(n => n.display ? -100 : 0)
			// .strength(n => n.link || n.componebnts ? 0 : -120)
			.strength(-100)
			.distanceMax(40),  n => !!n.link))
		.force("midpt_align", midpoint_aligning_force)
		// .force("bipartite", d3.forceLink(mk_bipartite_links(links)).id(l => l.id)
		// 	// .strength(l => 70/l.separation)
		// 	.distance(l => l.separation).iterations(3))
		.force("bipartite", custom_link_force(mk_bipartite_links(links)).id(l => l.id)
			// .distance(l => [l.separation*1, l.separation*1]).iterations(3))
			// .strength(1) 
			.distance(l => [l.separation / STRETCH_FACTOR, 
							l.separation * STRETCH_FACTOR]).iterations(3))
		// .force("nointersect", d3.forceCollide().radius(n => n.display ? n.w/2 : 0)
		// 		.strength(0.5).iterations(5))
		// .force("nointersect_old", d3.forceCollide().radius(
		// 			n => n.display ? n.w/2 : (n.link ? 5 : 0))
		// 		.strength(1).iterations(1))
		.force("nointersect", custom_collide_force()
			// .nodeFilter(n => !n.link)
			// .strength(0.7)
			.iterations(3))
		// .force("center", d3.forceCenter(canvas.width / 2, canvas.height / 2).strength(0.01))
		.on("tick", ontick)
		.stop();
	simulation.alphaDecay(0.05);
		
	setTimeout(reinitialize_node_positions, 10);

	
	function fresh_label(prefix="p") {
		existing = links.map( l => l.label);
		i = 1;
		while(existing.includes(prefix+i)) i++;
		return prefix+i;
	}
	function fresh_node_name(prefix="X") {
		// existing = N;
		existing = nodes.map(n => n.id);
		i = 1;
		while(existing.includes(prefix+i)) i++;
		return prefix+i;
	}
	function new_link(src, tgt, label, initial_ln_pos=[undefined,undefined]) {
		ensure_multinode(src);
		ensure_multinode(tgt);
		// simulation.nodes(nodes);
		align_node_dom();
		// simulation.force("anotherlink").links(parentLinks);
		
		let lobj = linkobject([label, [src,tgt]], links.length);
		links.push(lobj);
		// simulation.force("link").links(links);
		let ln = mk_linknode(lobj);
		ln.x = initial_ln_pos[0] == undefined ? ln.x : initial_ln_pos[0];
		ln.y = initial_ln_pos[1] == undefined ? ln.y : initial_ln_pos[1];

		linknodes.push(ln);
		simulation.nodes(nodes.concat(linknodes));
		simulation.force("bipartite").links(mk_bipartite_links(links));

		simulation.alpha(0.7).restart();
		
		return lobj;
	}
	function new_node(vname, x,y) {
		let ob = {
			id: vname, 
			x: x, y: y, vx: 0, vy:0,
			w : initw, h : inith,  display: true};
		nodes.push(ob);
		lookup[vname] = ob;
		align_node_dom();
		return ob;
	}
	function align_node_dom() {
		let nodedata = svg.selectAll(".node").data(nodes, n => n.id);
		let newnodeGs = nodedata.enter()
			.append("g")
			.classed("node", true);
			// .call(simulation.drag);
		newnodeGs.append("rect").classed("nodeshape", true);
		newnodeGs.append("text");		
		
		nodedata.exit().each(remove_node)
			.remove();
		
		nodedata = nodedata.merge(newnodeGs);
		nodedata.classed('expanded', n => n.expanded)
		nodedata.selectAll("rect.nodeshape")
			.attr('width', n => n.w).attr('x', n => -n.w/2)
			.attr('height', n => n.h).attr('y', n => -n.h/2)
			.attr('rx', 15);
		nodedata.selectAll("text").text(n => n.id);
		nodedata.filter( n => ! n.display).attr('display', 'none');
		
		if (typeof simulation != 'undefined') {
			simulation.nodes(nodes.concat(linknodes));
			simulation.force("bipartite").links(mk_bipartite_links(links));
			
			simulation.restart();
		}
	}
	function restyle_links() {
		let lndata = svg.selectAll(".linknode").data(linknodes, ln => ln.link.label);
		
		let newlnGs = lndata.enter().append("g").classed("linknode", true);
		newlnGs.append("text").classed("bg", true);
		newlnGs.append("text").classed("fg", true);

		lndata.exit().remove();
		
		lndata = lndata.merge(newlnGs);
		lndata.attr('transform', ln => "translate("+ ln.x+","+ln.y+")")
			.classed('selected', ln => ln.link.selected);
		lndata.selectAll("text").text(ln => ln.link.label);		
	}
	function restyle_nodes() {
		/*** Now for somedd SVG operations. ***/
		// let nodedata = 
		svg.selectAll(".node").data(nodes, n => n.id)
			// .attr("transform", n => "translate(" + lookup[n].x + ","+lookup[n].y +")")
			// .classed("selected", n => lookup[n].selected );
			.attr("transform", n => "translate(" + n.x + ","+ n.y +")")
			.classed("selected", n => n.selected );
	}
	function remove_node(n) {
		// console.log("removing node", n);
		for(let i = 0; i < links.length; i++) {
			l = links[i];
			// console.log("... |link ", l.label, l.source, l.target,
			 	// " --> remove? ",l.srcs.indexOf(n.id) >= 0 || l.tgts.indexOf(n.id) >= 0);
			// This test only works if this is the link object in a real force!!
			// if(l.source.id == n.id || l.target.id == n.id)
			// if(l.source == n.id || l.target == n.id)
			if(l.srcs.includes(n.id) || l.tgts.includes(n.id)) {
				remove_link(l);
				i--;
			}
		}
		// simulation.force("bipartite").links(mk_bipartite_links(links));
		
		let multis_to_remove = [];
		// for(let i = 0; i < parentLinks.length; i++) {
		// 	l = parentLinks[i];
		// 	if(l.source.id == n.id || l.target.id == n.id) {
		// 		parentLinks.splice(i,1);
		// 
		// 		cpt_idx = l.source.components.indexOf(n.id);
		// 		l.source.components.splice(cpt_idx,1);
		// 		// if( l.source.components.length == 0)
		// 		ensure_multinode(l.source.components);
		// 		multis_to_remove.push(l.source);
		// 		i--;
		// 	}
		// }
		for(let i = 0; i < nodes.length; i++) {
			let m = nodes[i];
			if(m == n) { // might already be gone, but make sure.
				nodes.splice(i,1); i--; continue;
			}
			if(m.components) { // remove n from other multinodes, 
				// ... or more accurately, delete them and create new,
				// smaller multi-nodes.
				let idx = m.components.indexOf(n.id)
				if(idx < 0) continue;
				m.components.splice(idx,1);
				ensure_multinode(m.components);
				multis_to_remove.push(m);
			}

		}
		delete lookup[n.id];
		multis_to_remove.forEach(remove_node);
	}
	function remove_link( l ) {
		// console.log("removing link ", l)
		var index = links.indexOf(l);
		if(index >= 0) {
			links.splice(index,1);
		}
		else if(l.label != 'templink')
			console.warn("link "+l.label+" not found for removal");
		index = linknodes.findIndex(ln => ln.link == l)
		if(index >= 0) {
			linknodes.splice(index, 1);
		}
		else if(l.label != 'templink')
			console.warn("linknode corresponding to "+l.label+" not found for removal");
	}
	
	function pickN(pt) {
		for(let objn of nodes) {
			adx = Math.abs(objn.x - pt.x);
			ady = Math.abs(objn.y - pt.y);

			if(adx <  objn.w/2 && ady < objn.h/2)
				return objn;
		}
	}
	function picksL(pt, l, extra_lw) {
		context.save();
		context.lineWidth = extra_lw + (l.lw | 2);
		let b = context.isPointInStroke(l.path2d, pt.x, pt.y);
		context.restore();
		return b;
	}
	function pickL(pt, extra_lw=6, return_ln=false) {
		context.save();
		// for(let l of links) {
		let l;
		for(let ln of linknodes) {
			l = ln.link;
			context.lineWidth = extra_lw + (l.lw | 2);
			if( context.isPointInStroke(l.path2d, pt.x, pt.y) ) {
				context.restore();
				return return_ln ? ln : l;
			}
		}
		context.restore();
	}
	
	d3.select(canvas).call(d3.drag()
			.container(canvas)
			.clickDistance(10)
			.subject(function(event) {
					// console.log("drag.subject passed : ", event)
					if (action.type == 'box-select') return true;
					// else if(action.type == '')
					if (mode == 'draw' && temp_link) return undefined;
					// else {

					let o = pickN(event);
					if(o) return o;
					let ln = pickL(event,6,true);
					if(ln) return ln;

					//  if in draw mode, 
					//  create new link source (empty srcs) beginning at target
					if(mode == 'draw') {
						let lo = {link: linkobject(['templink', [[],[]]]), x: event.x, y: event.y};
						return lo;
					}
					// }
				})
			.on("start", dragstarted)
			.on("drag", dragged)
			.on("end", dragended)
		);
	function dragstarted(event) {
		if(popup_process) clearTimeout(popup_process);
			
		if (action.type == 'box-select') {
			action.start = vec2(event);
			action.end = vec2(event);
			ontick();
			console.log("DRAGSTART", action)
		}
		else if(mode == 'move') {
			// if there are no other drag handlers currently firing.
			// apparently useful mostly in multi-touch scenarios.
			if (!event.active) simulation.alphaTarget(0.5).restart();
			if(event.subject.link)  {// it's a link
				event.subject.initial_offset = event.subject.offset;
			} else {  // if it's a node
				event.subject.fx = event.subject.x;
				event.subject.fy = event.subject.y;
			}
		}
		else if (mode == 'draw') {
			if(event.subject.link)  { // if it's an edge
				let l = event.subject.link;
				l.display = false; // don't display until it's cancelled or released. 
				temp_link = linkobject(['<TEMPORARY>', [l.srcs, ["<MOUSE>"].concat(l.tgts)]]);
				temp_link.based_on = l;
				temp_link.x = event.subject.x;
				temp_link.y = event.subject.y;
				// temp_link.unit
			} else { // drag.subject is a node.
				temp_link = linkobject(['<TEMPORARY>', [[event.subject.id], ["<MOUSE>"]]]);
			}
			ontick();
		}
	}
	function dragged(event) {
		// console.log(event);
		if (action.type == 'box-select') {
			action.end = vec2(event);
			ontick();
		}
		else if(mode == 'move') {
			if(event.subject.link)  { // if it's an edge
				// console.log(event);
				// event.subject.offset[0] += event.dx;
				// event.subject.offset[1] += event.dy;
			} else {// it's a node
				event.subject.fx = event.x;
				event.subject.fy = event.y;
			}
		} 
		else if (mode == 'draw') {
			// mouse_pt = vec2(event);
			lookup["<MOUSE>"] = {x: event.sourceEvent.x,
							y: event.sourceEvent.y,
							w:1,h:1
							// setting to negative 9 means the arrow is only shortened 1 pixel.
							// w: -9, h: -9
						};
			// ontick();
			redraw();
		}
	}
	function dragended(event) {
		if(action.type == 'box-select') {
			let [xmin,ymin,w,h] = corners2xywh(action.start, vec2(event));
			let xmax = xmin + w, 
				ymax = ymin + h;

			finalnode = pickN(event);

			for(let objn of nodes) {
				if (objn.x >= xmin && objn.x <= xmax && objn.y >= ymin && objn.y <= ymax || objn == finalnode) {
					objn.selected = event.sourceEvent.shiftKey ? !objn.selected : true;
					// console.log((objn.selected?"":"un")+"selecting  ", objn.id, event);
					// console.log((objn.selected?"":"un")+"selecting "+objn.id);
				} else {
					// console.log(event, event.sourceEvent.ctrlKey, event.sourceEvent.shiftKey);
					if(! event.sourceEvent.shiftKey && objn.selected ) {
						// 0 -> 0 (unselected); 1 -> 2 (demote primary selection); (2 -> 1)
						objn.selected = false;
					}
				}
			}
			restyle_nodes();
			
			// essentially copy paste of above, but with a .link because the 
			// event subject is a linknode, not a link (but .selected is in link).
			for(let ln of linknodes ){
				let l = ln.link;
				if (ln.x >= xmin && ln.x <= xmax && ln.y >= ymin && ln.y <= ymax) {
					l.selected = event.sourceEvent.shiftKey ? !l.selected : true;
				} else {
					if(! event.sourceEvent.shiftKey && l.selected ) {
						l.selected = false;
					}
				}
				//... plus also this code to close under node selection
				if(l.srcs.concat(l.tgts).every(n => lookup[n].selected)){
					l.selected = true;
				}
			}
			
			select_rect_start = null;
			select_rect_end = null;
			action = {};
			restyle_links();
			ontick();
		}
		if(mode == 'move') {
			if (!event.active){
				// simulation.alpha(1.2).alphaTarget(0).restart();	
				simulation.alphaTarget(0);
			} 
			
			if(event.subject.link)  { // if it's an edge
				// console.log("FINISH DRAG", event);
				// event.subject.offset = [ 
				// 		event.subject.initial_offset[0] + event.,
				// 		event.subject.initial_offset[1] + event.dy ]
			} else {// it's a node	
				if(!event.subject.expanded) {
					event.subject.fx = null;
					event.subject.fy = null;
				}
			}
		}
		else if (mode == 'draw' && temp_link) {
			let newtgts = [], newsrcs = [];
			
			let pickobj = pickN(event.sourceEvent);
			if( pickobj ) {
				// disable self-edges (for now) --- they're very annoying and easy to make by accident
				if((temp_link.srcs.length == 1) && (temp_link.srcs[0] == pickobj.id)) {
					temp_link = null;
					console.log("aborting; no self loop");
					redraw();
					return;
				}
				
				newtgts.push(pickobj.id);
			} else { // no pick object. 
				pickl = pickL(event.sourceEvent, 25);
				if(pickl) {
					if(pickl == temp_link.based_on){
						temp_link.based_on.display = true;
						temp_link = null;
						ontick(); 
						return;
					}
					newsrcs.push(...pickl.srcs);
					newtgts.push(...pickl.tgts.filter( n => !newtgts.includes(n)));
					remove_link(pickl);
				} else {
					// create new edge (Or abandon?)
					pickobj = new_node(fresh_node_name(), event.x, event.y);
					if(!newtgts.includes(pickobj.id)) newtgts.push(pickobj.id);
					
					
					// don't bother making a link if it was just a click;
					// just make the new node.
					// console.log(event, temp_link, action);
					if(temp_link.srcs.length == 0 && mag(subv(vec2(event), vec2(temp_link))) <= 20) {
						temp_link = null; ontick();	return;
					}
				}
			}
			if(event.subject.link) { // event source was a link
				newtgts.push(...event.subject.link.tgts.filter( n => !newtgts.includes(n)));
				remove_link(event.subject.link);
			}


			// let newtgts = [pickobj.id] // do I maybe want to do this at end?
			newsrcs.push(... temp_link.srcs.filter( n => !newsrcs.includes(n)));
			new_link(newsrcs, newtgts, fresh_label(), [temp_link.x, temp_link.y]);
			simulation.alpha(0.5).alphaTarget(0).restart();
			
			temp_link = null;
			ontick();
		}
	}
	
	function set_mode(mode) {
		$("#drag-mode-toolbar button[data-mode='"+mode+"']").click();
	}
	
	canvas.addEventListener("dblclick", function(e) {
		let obj = pickN(e), link = pickL(e);
		if(obj) { // rename selected node
			// EXPANDING CODE
			if(!obj.expanded) {
				simulation.stop();
				obj.expanded = true;
				obj.old_wh = [obj.w, obj.h];
				// [obj.w, obj.h] = [550,250];
				[obj.w, obj.h] = [200,150];
				[obj.fx, obj.fy] = [obj.x, obj.y];
				simulation.alpha(2).alphaTarget(0).restart();
				
				for(let ln of linknodes) {
					// if l.srcs or l.tgts includes n,
					// then set strength to zero?
					// set distance?
				}
			}
			else {
				obj.expanded = false;
				[obj.w, obj.h] = obj.old_wh ? obj.old_wh : [initw,inith];
				delete obj.fx
				delete obj.fy;
				simulation.alpha(2).alphaTarget(0).restart();
			}
			align_node_dom();
			
			
			//RENAMING CODE
			// let name = promptForName("Enter New Variable Name", obj.id, nodes.map(n=>n.id));
			// if(!name) return;
			// 
			// let replacer = nid => (nid == obj.id) ? name : nid;
			// //TODO this will leave parentLinks in the dust...
			// for(let l of links) {
			// 	l.srcs = l.srcs.map(replacer);
			// 	l.tgts = l.tgts.map(replacer);
			// 	l.source = l.srcs.join(",");
			// 	l.target = l.tgts.join(",");
			// }
			// delete lookup[obj.id];
			// obj.id = name;
			// lookup[name] = obj;
			// align_node_dom();
		} else if(link) { // rename selected cpd
			
			
		} else { // nothing selected; create new variable here.
			setTimeout(function() {
				let name = promptForName("Enter A Variable Name", fresh_node_name(), nodes.map(n=>n.id));
				if(!name) return;
				
				newtgt = new_node(name, e.x, e.y);
				if(temp_link) {
					// todo: fold out this functionality, shared with click below.
					new_tgts = temp_link.tgts.slice(1);
					new_tgts.push(newtgt.id);
					new_link(temp_link.srcs, new_tgts, fresh_label(), [temp_link.x, temp_link.y]);
					temp_link = null;
				}
				ontick();
			}, 10);
		}		
		// if(e.ctrlKey || e.metaKey) {
		// }
	});
	canvas.addEventListener("click", function(e) {
		// ADD NEW NODE
		// if(e.ctrlKey || e.metaKey) {
		if( temp_link ) {
			let newtgt = pickN(e);
			if(!newtgt && mode == 'draw') {
				newtgt = new_node(fresh_node_name(), e.x, e.y);
			}
			if(newtgt) {
				if(!e.shiftKey) {
					new_tgts = temp_link.tgts.slice(1);
					new_tgts.push(newtgt.id);
					new_link(temp_link.srcs, new_tgts, fresh_label(), [temp_link.x, temp_link.y]);
					temp_link = null;
				}
				else {
					temp_link.tgts.push(newtgt.id);
				}
			}	
		} else if(action.type == 'move') {
			mouse_end = vec2(lookup['<MOUSE>']);
			
			action.targets.forEach(n => {
				[n.x, n.y] = addv(n.old_pos, mouse_end, scale(action.mouse_start, -1)); 
				delete n.old_pos;
			});
			
			function adjust_seps(ln, n, nsibls, isloop) {
				// ln.sep[n] = mag(subv(vec2(ln), vec2(lookup[n])));
				let p = vec2(ln), 
					q = vec2(lookup[n]),
					wh = [lookup[n].w, lookup[n].h];
				let cur_sep = mag(subv(sqshortened_end(q,p,[ln.w,ln.h]),
									 sqshortened_end(p,q, wh) ));
				let cur_sep_want = ln.sep && ln.sep[n]? ln.sep[n] :
				 	default_separation(nsibls,isloop)
				
				if(cur_sep > cur_sep_want * STRETCH_FACTOR ||
					 	cur_sep < cur_sep_want / STRETCH_FACTOR) 
					ln.sep[n] = cur_sep;
			}
			
			for(let ln of linknodes) {
				if(action.targets.includes(ln)) {
					ln.sep = {}
					// ## TEMPORARILY COMMENTED OUT; KEEP SEPS SAME
					// for(let n of ln.link.srcs) 
					// 	adjust_seps(ln, n, ln.link.srcs.length, ln.link.tgts.includes(n))
					// 
					// for(let n of ln.link.tgts) 
					// 	adjust_seps(ln, n, ln.link.tgts.length, ln.link.srcs.includes(n))
				}
			}
			simulation.force("bipartite").links(mk_bipartite_links(links));

			// for(let n of action.targets) {
			// 
			// }
			action = {};
			restyle_nodes();
			
		} else if(mode == 'move') { // selection in manipulate mode
			let obj = pickN(e), link = pickL(e);
			
			if( obj || link)  {
				if( !e.shiftKey)  {
					nodes.forEach( n => {if(n != obj) n.selected=false;} );
					links.forEach( l => {if(l != link) l.selected=false;} );
				}
				
				if(obj) {
					// console.log("toggling ", obj.id, e);
					obj.selected = !obj.selected;
					for(let l of links) {	
						if(l.srcs.concat(l.tgts).every(n => lookup[n].selected)) {
							l.selected = true;
						}
					}
				} 
				if(link) link.selected = !link.selected;
				
				
				restyle_nodes();
				restyle_links();
			}
		}
		// else if(mode == 'select'){
		// 	let link = pickL(e);
		// 	if(link) link.selected = !link.selected;
		// 	// console.log("[Click] " + (link.selected?"":"un")+"selecting  ", link.label, e);
		// 	redraw();
		// }
	});
	window.addEventListener("keydown", function(event){
		// console.log(event);
		
		if(event.key == 'Escape'){
			if ( temp_link ) {
				if(temp_link.based_on ) 
					temp_link.based_on.display = true;
				
				temp_link = null;
				redraw();
			}
			else if( action.type == 'move') {
				action.targets.forEach(n => {
					[n.x, n.y] = n.old_pos;
					delete n.old_pos;
				});
			}
			action = {};
			ontick();
		}
		else if (event.key == 'a') {
			let all_selected = true;
			for(let s of nodes.concat(links)) {
				if(! s.selected) 
					all_selected = false;
				s.selected = true;
			}
			if(all_selected) {
				for(let s of nodes.concat(links))
					s.selected = false;
			}
			restyle_nodes();
			restyle_links();
			redraw();
		}
		else if (event.key.toLowerCase() == 'b') {
			// $("#drag-mode-toolbar button[data-mode='select']").click();
			// set_mode('select');
			
			action = {
				type: "box-select",
				end : null,
				start : null
			}
		}
		else if (event.key.toLowerCase() == 't') {
			// start creating arrows.
			// 1. Create new arrow from selection at tail
			src = nodes.filter( n => n.selected ).map( n => n.id );
			// lab = fresh_label();
			// temp_link = new_link(src, ['<MOUSE>'], "<TEMPORARY>");
			temp_link = linkobject(['<TEMPORARY>', [src, ["<MOUSE>"]]], undefined)
			if( src.length == 0) {
				temp_link.x = lookup["<MOUSE>"].x
				temp_link.y = lookup["<MOUSE>"].y
			}
			// set_mode('draw');
			// links.push(temp_link);
		}
		else if (event.key == ' ') {
			event.preventDefault();
			// simulation.alphaTarget(0.05).restart();
			simulation.alpha(2).alphaTarget(0).restart();
			
			if(mode == 'move') {
			}
			if(mode == 'select') {
				// TODO shift selection to backup selection (red color)
			}
		}
		else if (event.key.toLowerCase() == 'x') {
			simulation.stop();
			nodes = nodes.filter(n => !n.selected);
			// for(let i = 0; i < )
			links_to_remove = links.filter( l => l.selected);
			links_to_remove.map(remove_link);
			align_node_dom();
		}
		else if (event.key == 'd') {
			set_mode("draw");
		}
		else if (event.key == 'm') {
			set_mode("move");
		}
		else if (event.key == "g") {
			simulation.stop();
			// move selection with mouse
			
			action = {
				type : "move", 
				mouse_start : vec2(lookup["<MOUSE>"]),
				targets: nodes.filter(n => n.selected).concat(linknodes.filter(ln => ln.link.selected)) 
			}
			
			action.targets.forEach( n => {
				n.old_pos = vec2(n);
			});
		}
		else if (event.key == "s") {
			
		}

	});
	canvas.addEventListener("wheel", function(e) {
		// console.log("canvas", e.wheelDelta );
		// lover = pickL(e, width=10);
		
		//# code to change LINE WIDTH
		// if(lover.lw == undefined) lover.lw=2;
		// lover.lw = (lover.lw + sgn(e.wheelDelta) );
		
		
		ontick();
		// console.log(lover);
	});
	window.addEventListener("mousemove", function(e) {
		// mouse_pt = [e.x, e.y];
		lookup["<MOUSE>"] = {x : e.x, y: e.y, w:0,h:0};
		if(temp_link) redraw();
		
		if(popped_up_link && !picksL(e, popped_up_link, 10)) {
			delete popped_up_link.lw;
			popped_up_link = null;
			ontick();
		}
		
		if(popup_process) clearTimeout(popup_process);
	
		if( !popped_up_link) {
			popup_process = setTimeout(function() {
				let l = pickL(e, 10);
				popped_up_link = l;

				if(l) {
					l.lw = 5;
					ontick();
				}
			}, 100);
		}

		// if ( mode == 'move'  && action) {
		if(action.type == 'move') {
			action.targets.forEach(n => {
				[n.x, n.y] = addv(n.old_pos, vec2(e), scale(action.mouse_start, -1)); 
			});
			restyle_nodes();
			// midpoint_aligning_force(1);
			ontick();
			// TODO move selection, like ondrag below
		}
	})
});
