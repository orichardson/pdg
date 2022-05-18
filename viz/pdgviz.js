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
// 	nodes : ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6'],
//  	hedges: {
// 		'p271': [['X2', 'X6', 'X4'], ['X0']],
// 		'p272': [['X0', 'X5'], ['X1']],
// 		'p273': [['X1'], ['X2']],
// 		'p274': [['X2'], ['X3']],
// 		'p275': [['X3', 'X5'], ['X4']],
// 		'p276': [['X2', 'X3', 'X4'], ['X5']],
// 		'p277': [['X2'], ['X6']]}
// };

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
const OPT_DIST = {0 : 35, 1:50, 2:70, 3:100, 4: 150, 5: 180, 6: 180};

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
	
	
	let mode = $('#drag-mode-toolbar button.active').attr('data-mode');
	console.log(mode);
	
	$('#drag-mode-toolbar button').on('click', function() {
		$('#drag-mode-toolbar button').removeClass("active");
		$(this).addClass('active');
		mode = $(this).attr('data-mode');
		console.log('new mode: ', mode);
	});


	var select_rect_end,  select_rect_start;
	var mouse_pt = [0,0];
	

	let context = canvas.getContext("2d");
	lookup = [];
	// TODO make this a let later
	nodes = N.map( function(varname) {
		let ob = {"id": varname, "w" : initw, "h": inith, "display": true};
		lookup[varname] = ob;
		return ob;
	});
	// window.nodes = nodes;
	let parentLinks = [];
	// TODO let, later
	// let link_nodes = [];
	
	function ensure_multinode(multi) {
		s = multi.join(',')
		if( ! nodes.find(n => n.id == s )) {
			let ob = {id:s, w:6, h:6, display: false,
				components: multi,
			 	vx:0.0, vy:0.0};
			
			if( multi.length > 0 ) 
				[ob.x, ob.y] = avgpos(...multi); // defined below.
		
			nodes.push(ob);
			lookup[s] = ob;
			multi.forEach(n =>
				parentLinks.push({"source" : s, "target" : n}) );
		};
	}
	for (label in ED) {
		for(var multi of ED[label]) {
			ensure_multinode(multi);
		}
	}


	let nodedata = svg.selectAll(".node").data(nodes, n => n.id);
	let gnode = nodedata.enter().append("g").classed("node", true);
	gnode.append("rect")
		.classed("nodeshape", true)
		.attr('width', n => n.w).attr('x', n => -n.w/2)
		.attr('height', n => n.h).attr('y', n => -n.h/2)
		.attr('rx', 15);
	gnode.append("text").text(n => n.id);
	gnode.filter( n => ! n.display).attr('display', 'none')

	//##  Next, Updating + Preparing shapes for drawing.  
	//##  But first, some helpful functions.
	function avgpos( ... nodenames ) {
		// if ( nodenames[0] == "<MOUSE>")
		// 	return mouse_pt;
		return   [ d3.mean(nodenames.map(v => lookup[v].x)),
				   d3.mean(nodenames.map(v => lookup[v].y)) ];
	}
	function compute_link_shape( src, tgt, midpt=undefined, return_mid=false) {
		// let srcnode = lookup[src.join(",")];
		// let avgsrc = vec2(srcnode);
		// if( src.length > 0 ) {
		// 	avgsrc = avgpos(...src);
		// }
		let avgsrc = src.length==0 ? vec2(lookup['']) : avgpos(...src);
		
		// let tgtnode = lookup[tgt.join(",")];
		// let avgtgt = vec2(tgtnode);
		// if( tgt.length == 0 ) {
		// 	avgtgt = avgpos(...tgt);
		// }
		let avgtgt = tgt.length==0 ? vec2(lookup['']) : avgpos(...tgt);

		// let mid = [ 0.4*avgsrc[0] + 0.6*avgtgt[0], 0.4*avgsrc[1] + 0.6*avgtgt[1] ];
		// let mid = midpt ? midpt : 
		let mid = [ 0.4*avgsrc[0] + 0.6*avgtgt[0], 0.4*avgsrc[1] + 0.6*avgtgt[1] ];
		// console.log('ho', avgsrc,avgtgt, mid);
		function shortener(s) {
			return sqshortened_end(mid, vec2(lookup[s]), [lookup[s].w, lookup[s].h], 10);
		}
		let avgsrcshortened = src.length == 0 ? 
			shortener("") : scale(addv(... src.map(shortener)), 1 / src.length);
		let avgtgtshortened = tgt.length == 0 ?
			shortener("") : scale(addv(... tgt.map(shortener)), 1 / tgt.length);
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
					0.2*midearly[0] + startpt[0]*(0.8) + delta[0] * 0.5,
					0.2*midearly[1] + startpt[1]*(0.8) + delta[1] * 0.5,
					.8*avgsrcshortened[0] + true_mid[0]*(0.2) + delta[0],
					.8*avgsrcshortened[1] + true_mid[1]*(0.2) + delta[1],
					// lookup[s].x, lookup[s].y,
					mid[0], mid[1]);

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
			lpath.lineTo(...endpt);
			let [ar0, ar1, armid0, armid1] = arrowpts(mid, endpt, 10);
			lpath.moveTo(...endpt);
			lpath.quadraticCurveTo(armid0[0], armid0[1], ar0[0], ar0[1]);
			lpath.moveTo(...endpt);
			lpath.quadraticCurveTo(armid1[0], armid1[1], ar1[0], ar1[1]);
		});
		if(return_mid) return [lpath, true_mid];
		return lpath;
	}
	function ontick() {
		// for (label in ED) {
		// 	 let [src, tgt] = ED[label];
		// for (let l of links) {
		// 	l.path2d = compute_link_shape(l.srcs,l.tgts);
		// }
		for (let ln of linknodes) {
			let l = ln.link;
			[l.path2d, ln.true_mid] = compute_link_shape(l.srcs, l.tgts, vec2(ln), true);
		}
		
		nodes.forEach(function(n) {
			//updating
			n.x = clamp(n.x, n.w/2, canvas.width - n.w/2);
			n.y = clamp(n.y, n.h/2, canvas.height - n.h/2);
		});
		
		restyle_nodes();
		redraw();
	}
	function redraw() {
		context.save();
		context.clearRect(0, 0, canvas.width, canvas.height);
		
		context.lineWidth = 1.5;
		context.strokeStyle = "black";
		// context.setLineDash([]);

		for( let l of links) {
			context.lineWidth = 5;
			context.strokeStyle = "white";
			context.stroke(l.path2d);
			
			context.lineWidth = 1.5;
			context.strokeStyle = "black";
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
			context.lineWidth = 3;
			context.strokeStyle = "white";
			context.stroke( compute_link_shape(temp_link.srcs, temp_link.tgts ))
			
			context.lineWidth = 1.5;
			context.strokeStyle = "black";
			context.stroke( compute_link_shape(temp_link.srcs, temp_link.tgts ))
		}
		
		// Draw Selection Rectangle
		context.globalAlpha = 0.2;
		if( mode == "select" && select_rect_start && select_rect_end ) {
			// console.log(...corners2xywh(select_rect_start, select_rect_end))
			// context.save();
			context.fillStyle="orange";
			
			// context.fillRect(select_rect_start.x, select_rect_start.y, select_rect_end.x, select_rect_end.y);
			// let [xmin,ymin,w,h] = corners2xywh(select_rect_start, select_rect_end);
			context.fillRect(...corners2xywh(select_rect_start, select_rect_end));
			// context.stroke();
			// context.restore();
		}
		
		/// Draw the invisible product nodes + make sure no node goes off screen.
		context.globalAlpha = 0.5;
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
		
		//draw the linknodes 
		// linknodes.forEach(function(n) {
		// 	context.beginPath();
		// 
		// 	if( n.selected )
		// 		context.strokeStyle="#1AE";
		// 	else context.strokeStyle="#A4C";
		// 
		// 	context.moveTo(n.x, n.y);
		// 	context.arc(n.x, n.y, 8, 0, 2 * Math.PI);
		// 	context.stroke();				
		// });
		context.globalAlpha = 1;
		context.restore();
	}
	
	function linkobject([label, [src,tgt]], i) {
		// return { "source" : src.join(","), "target" : tgt.join(","), "index": i};
		return {
			source: src.join(","), 
			target: tgt.join(","), 
			index: i, 
			label: label,
			display: true,
			srcs : src,
			tgts : tgt
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
			vx: 0, vy:0, w : 10, h : 10,  display: false};
		return ob;
	}
	links = Object.entries(ED).map(linkobject);
	
	window.avgpos = avgpos;
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
		for (let ln of linknodes) {
			let l = ln.link;
			[l.path2d, ln.true_mid] = compute_link_shape(l.srcs, l.tgts, vec2(ln), true);
			// ln.x += (mid[0] - ln.x) * 0.25;
			// ln.y += (mid[1] - ln.y) * 0.25;
			ln.vx += (ln.true_mid[0] - ln.x) * 0.35 *alpha;
			ln.vy += (ln.true_mid[1] - ln.y) * 0.35 *alpha;
		}
	}

	linknodes = links.map(mk_linknode)
	function mk_bipartite_links(links){
		bipartite_links = []
		for( let l of links) {
			let lname = "ℓ" + l.label;
			
			// let loops = l.srcs.filter(n => l.tgts.includes(n));
			
			// l.srcs.map( s =>  {source:s, target:lname})
			for( let s of l.srcs) {
				bipartite_links.push({ 
					source: s, target: lname, 
					nsibls: l.srcs.length, 
					isloop: l.tgts.includes(s)
				});
			}
			for( let t of l.tgts) {
				bipartite_links.push({
					source: lname, target: t, 
					nsibls: l.tgts.length, 
					isloop: l.srcs.includes(t) 
				});
			}
		}
		return bipartite_links;
	}
	window.mk_bipartite_links = mk_bipartite_links;
	// TODO: Make this center force change on resize.
	simulation = d3.forceSimulation(nodes.concat(linknodes))
	// simulation = d3.forceSimulation(nodes)
		
		//// .force("charge", d3.forceManyBody().strength( -100))
		// .force("link", d3.forceLink(links).id(l => l.id)
		// 	.strength(1).distance(110).iterations(3))
		// .force("anotherlink", d3.forceLink(parentLinks).id(n=>n.id)
		// 		.strength(0.3).distance(40).iterations(2))
		.force("avgpos_align", multi_avgpos_alignment_force)
		.force("charge", d3.forceManyBody()
			// .strength(n => n.display ? -100 : 0))
			.strength(n => n.link ? 0 : -120))
		.force("midpt_align", midpoint_aligning_force)
		.force("bipartite", d3.forceLink(mk_bipartite_links(links)).id(l => l.id)
			.strength(1).distance(l => {
				let optdist = (l.nsibls in OPT_DIST ? OPT_DIST[l.nsibls] : 30*l.nsibls) + sgn(l.isloop)*50;
				// if( l.link.)
				// console.log("Evaluating Distance Accessor.",optdist);
				return optdist;
			}).iterations(3))
		// .force("nointersect", d3.forceCollide().radius(n => n.display ? n.w/2 : 0)
		// 		.strength(0.5).iterations(5))
		.force("nointersect", d3.forceCollide().radius(
					n => n.display ? n.w/2 : (n.link ? 10 : 0))
				.strength(0.5).iterations(5))
		.force("center",
			d3.forceCenter(canvas.width / 2, canvas.height / 2).strength(0.1))
		.on("tick", ontick)
		.stop();
	simulation.alphaDecay(0.05);
		
	setTimeout(function(){
		for (const node of nodes) {
			node.x = node.x * 10.8 + canvas.width/2;
			node.y = node.y * 10.8 + canvas.height/2;
		} ontick();} , 1);
	setTimeout(() => { simulation.restart(); }, 100);


	function pick(pt) {
		for(let objn of nodes) {
			adx = Math.abs(objn.x - pt.x);
			ady = Math.abs(objn.y - pt.y);

			if(adx <  objn.w/2 && ady < objn.h/2)
				return objn;
		}
	}

	d3.select(canvas)
	    .call(d3.drag()
	        .container(canvas)
	        .subject(function(event) {
					if (mode == 'select') return true;
					if (mode == 'draw' && temp_link) return undefined;
					else return pick(event);
				})
	        .on("start", dragstarted)
	        .on("drag", dragged)
	        .on("end", dragended)
		);
	
	function set_mode(mode) {
		$("#drag-mode-toolbar button[data-mode='"+mode+"']").click();
	}
	function fresh_label(prefix="p") {
		// existing = Object.keys(ED);
		existing = links.map( l => l.label)
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
	function new_link(src, tgt, label) {
		console.log("New Link: ", src, tgt, label);
		ensure_multinode(src);
		ensure_multinode(tgt);
		// simulation.nodes(nodes);
		align_node_dom();
		// simulation.force("anotherlink").links(parentLinks);
		
		ED[label] = [src, tgt];
		lobj = linkobject([label, [src,tgt]], links.length);
		console.log(lobj);
		console.log(mk_linknode(lobj));
		links.push(lobj);
		// simulation.force("link").links(links);
		// linknodes = links.map(mk_linknode);
		linknodes.push(mk_linknode(lobj));
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
	}
	function align_node_dom() {
		nodedata = svg.selectAll(".node").data(nodes, n => n.id);
		// nodedata = svg.selectAll(".node").data(nodes.concat(linknodes), n => n.id);

			// .enter().append("g").classed("node", true);
		newnodeGs = nodedata.enter()
			.append("g")
			.classed("node", true);
			// .call(simulation.drag);
		newnodeGs.append("rect")
			.classed("nodeshape", true)
			.attr('width', n => n.w).attr('x', n => -n.w/2)
			.attr('height', n => n.h).attr('y', n => -n.h/2)
			.attr('rx', 15);
		newnodeGs.append("text").text(n => n.id);
		newnodeGs.filter( n => ! n.display).attr('display', 'none');
		
		// dunno if this does anything yet
		nodedata.exit().each(remove_node)
			.remove();
		
		// linknodes = links.map(mk_linknode);
		simulation.nodes(nodes.concat(linknodes));
		simulation.force("bipartite").links(mk_bipartite_links(links));

		// simulation.nodes(nodes);
		simulation.restart();
	}
	function restyle_nodes() {
		/*** Now for some SVG operations. ***/
		nodedata = svg.selectAll(".node").data(nodes, n => n.id);
		nodedata
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
		console.log("removing link ", l)
		var index = links.indexOf(l);
		if(index >= 0) {
			links.splice(index,1);
		}
		else console.warn("link "+l.label+" not found for removal");
		index = linknodes.findIndex(ln => ln.link == l)
		if(index >= 0) {
			linknodes.splice(index, 1);
		}
		else console.warn("linknode corresponding to "+l.label+" not found for removal");

		delete ED[l.label];
	}
	window.remove_link = remove_link;
		
	canvas.addEventListener("dblclick", function(e) {
		if (true) { // mode guard later 
			// TODO guards: not already on top of a node. Picks none. 
			// new node.
			
			if(!pick(e))
			setTimeout(function() {
				let name = window.prompt("Enter A Variable Name", fresh_node_name());
				existing = nodes.map(n => n.id);
				if(name) {
					if(existing.includes(name)) {
						window.alert(`Variable name ${name} already taken.`);
					} else {
						new_node(name, e.x, e.y);
						ontick();
					}
				}
			}, 10);
		}
		
		if(e.ctrlKey || e.metaKey) {
		}
	});
	canvas.addEventListener("click", function(e) {
		// ADD NEW NODE
		// if(e.ctrlKey || e.metaKey) {
		console.log("click event");
	
		if( temp_link ) {
			newtgt = pick(e);
			if(newtgt) {
				if(!e.shiftKey) {
					new_tgts = temp_link.tgts.slice(1);
					new_tgts.push(newtgt.id);
					new_link(temp_link.srcs, new_tgts, fresh_label());
					temp_link = null;
				}
				 else {
					temp_link.tgts.push(newtgt.id);
				}
			}	
		} else if(mode == 'move') {
			obj = pick(e)
			if(obj) {
				if(!e.shiftKey)
					nodes.forEach(n => {if(n.id != obj.id) n.selected=false;});

				// if(obj.selected) {
				obj.selected = !obj.selected;
				restyle_nodes();
		 	}
		}
		// }
	});

	var temp_link = null;
	
	window.addEventListener("keydown", function(event){
		console.log(event);
		
		if(event.key == 'Escape'){
			if ( temp_link ) {
				temp_link = null;
				redraw();
			}
			else {
			}
		}
		else if (event.key == 'b') {
			// $("#drag-mode-toolbar button[data-mode='select']").click();
			set_mode('select');
		}
		else if (event.key == 't') {
			// start creating arrows.
			// 1. Create new arrow from selection at tail
			src = nodes.filter( n => n.selected ).map( n => n.id );
			// lab = fresh_label();
			// temp_link = new_link(src, ['<MOUSE>'], "<TEMPORARY>");
			temp_link = linkobject(['<TEMPORARY>', [src, ["<MOUSE>"]]], undefined)
			// set_mode('draw');
			// links.push(temp_link);
		}
		else if (event.key == ' ') {
			// simulation.alphaTarget(0.05).restart();
			simulation.alpha(1).alphaTarget(0).restart();
			
			if(mode == 'move') {
			}
			if(mode == 'select') {
				// TODO shift selection to backup selection (red color)
			}
		}
		else if (event.key == 'x') {
			// nodedata = svg.selectAll('.node').data(nodes, n => n.id);
			// nodedata.
			// 	filter(n => n.selected).remove();
			// links = links.filter(l => l.)
			simulation.stop();
			nodes = nodes.filter(n => !n.selected);
			align_node_dom();
		}
		else if (event.key == 'd') {
			set_mode("draw");
		}
		else if (event.key == 'm') {
			set_mode("move");
		}

	});
	
	window.addEventListener("mousemove", function(e) {
		// mouse_pt = [e.x, e.y];
		lookup["<MOUSE>"] = {x : e.x, y: e.y, w:5,h:5};
		if(temp_link) redraw();
		
		if ( mode == '' ) {
			// TODO make it so that, after t is pressed, there's
			// a phantom node that follows the cursor, until click.
			// nodes
		}
		if ( mode == 'move' /* && gpressed */ ) {
			// TODO move selection, like ondrag below
		}
	})


	function dragstarted(event) {
		if(mode == 'move') {
			if (!event.active) simulation.alphaTarget(0.5).restart();
			event.subject.fx = event.subject.x;
			event.subject.fy = event.subject.y;
		}
		else if (mode == 'select') {
			select_rect_start = vec2(event);
			select_rect_end = vec2(event);
			ontick();
		}
		else if (mode == 'draw') {
			console.log(event.subject);
			temp_link = linkobject(['<TEMPORARY>', [[event.subject.id], ["<MOUSE>"]]], undefined);
			ontick();
		}
	}

	function dragged(event) {
		if(mode == 'move') {
			event.subject.fx = event.x;
			event.subject.fy = event.y;
		} 
		else if (mode == 'select') {
			select_rect_end = vec2(event);
			ontick();
		}
		else if (mode == 'draw') {
			ontick();
			// mouse_pt = vec2(event);
			lookup["<MOUSE>"] = {x: event.x, y:event.y, w:5,h:5};
		}
	}

	function dragended(event) {
		if(mode == 'move') {
			if (!event.active) simulation.alphaTarget(0);
			event.subject.fx = null;
			event.subject.fy = null;
		}
		else if (mode == 'select') {
			let [xmin,ymin,w,h] = corners2xywh(select_rect_start, select_rect_end);
			let xmax = xmin + w, 
				ymax = ymin + h;

			for(let objn of nodes) {
				if (objn.x >= xmin && objn.x <= xmax && objn.y >= ymin && objn.y <= ymax) {
					objn.selected = true;
				} else {
					// console.log(event, event.sourceEvent.ctrlKey, event.sourceEvent.shiftKey);
					if(! event.sourceEvent.shiftKey && objn.selected ) {
						// 0 -> 0 (unselected); 1 -> 2 (demote primary selection); (2 -> 1)
						objn.selected = false;
					}
				}
			}
			finalnode = pick(event);
			if(finalnode) finalnode.selected = true;
			restyle_nodes();
			
			select_rect_start = null;
			select_rect_end = null;
			ontick();
		} else if (mode == 'draw') {
			pickobj = pick(event);
			if(pickobj) {
				new_link(temp_link.srcs, [pickobj.id], fresh_label());
			}
			temp_link = null;
			ontick();
		}
	}
});



function  corners2xywh(start, end) {
	xmin = Math.min(start[0], end[0]);
	xmax = Math.max(start[0], end[0]);
	ymin = Math.min(start[1], end[1]);
	ymax = Math.max(start[1], end[1]);
	// return [xmin,xmax,ymin,ymax];
	return [xmin, ymin, xmax-xmin, ymax-ymin];
}

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
	if(delta[0] == 0 && delta[1] == 0) {
		return addv(from, [w/2, h/2]);
	}
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
