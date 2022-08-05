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
			pdg.tick();
		}
		// pdg.tick();
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
	
	
	pdg = PDGView(hypergraph)
	pdgs =  [ pdg ]
	
	let mouse = { w : 0, h: 0 };
	pdg.lookup["<MOUSE>"] = mouse;
		
	$('#save-button').click(function(e){
		download_JSON(pdg.state, 'hypergraph');
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
			// load_hypergraph(ob);
			pdg.load(ob);
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
	function redraw() {
		context.save();
		context.clearRect(0, 0, canvas.width, canvas.height);
		
		context.lineWidth = 1.5;
		context.strokeStyle = "black";

		context.lineCap = 'round';
		// context.setLineDash([]);
		
		for( let M of pdgs ) {
			M.draw(context);
		}

		
		if(temp_link) {
			let midpt = (temp_link.x == undefined) ? undefined : vec2(temp_link);
			let tlpath = pdg.compute_link_shape(temp_link.srcs, temp_link.tgts, midpt);
			
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
		
		
	}
	pdg.repaint_via(redraw);
	
	d3.select(canvas).call(d3.drag()
			.container(canvas)
			.clickDistance(10)
			.subject(function(event) {
					// console.log("drag.subject passed : ", event)
					if (action.type == 'box-select') return true;
					// else if(action.type == '')
					if (mode == 'draw' && temp_link) return undefined;
					// else {

					let o = pdg.pickN(event);
					if(o) return o;
					let ln = pdg.pickL(event,6,true);
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
			pdg.tick();
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
			pdg.tick();
		}
	}
	function dragged(event) {
		// console.log(event);
		if (action.type == 'box-select') {
			action.end = vec2(event);
			pdg.tick();
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
			// pdg.lookup["<MOUSE>"] = {x: event.sourceEvent.x,
			// 				y: event.sourceEvent.y,
			// 				w:1,h:1
			mouse.x = event.sourceEvent.x;
			mouse.y = event.sourceEvent.y;
							// setting to negative 9 means the arrow is only shortened 1 pixel.
							// w: -9, h: -9
						// };
			// pdg.tick();
			redraw();
		}
	}
	function dragended(event) {
		if(action.type == 'box-select') {
			action.end = vec2(event);
			action.shift = event.sourceEvent.shiftKey;
			pdg.handle(action)
			
			select_rect_start = null;
			select_rect_end = null;
			action = {};
			redraw();
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
			pdg.handle({
				type : "edge-stroke",
				temp_link : temp_link,
				endpt: {x : event.sourceEvent.x, y : event.sourceEvent.y},
				// source_link : event.subject.link
			});
			temp_link = null;
		}
	}
	
	function set_mode(mode) {
		$("#drag-mode-toolbar button[data-mode='"+mode+"']").click();
	}
	
	canvas.addEventListener("dblclick", function(e) {
		let obj = pdg.pickN(e), link = pdg.pickL(e);
		if(obj) { // rename selected node
			// EXPANDING CODE
			// if(!obj.expanded) {
			// 	simulation.stop();
			// 	obj.expanded = true;
			// 	obj.old_wh = [obj.w, obj.h];
			// 	// [obj.w, obj.h] = [550,250];
			// 	[obj.w, obj.h] = [200,150];
			// 	[obj.fx, obj.fy] = [obj.x, obj.y];
			// 	simulation.alpha(2).alphaTarget(0).restart();
			// 
			// 	for(let ln of linknodes) {
			// 		// if l.srcs or l.tgts includes n,
			// 		// then set strength to zero?
			// 		// set distance?
			// 	}
			// }
			// else {
			// 	obj.expanded = false;
			// 	[obj.w, obj.h] = obj.old_wh ? obj.old_wh : [initw,inith];
			// 	delete obj.fx
			// 	delete obj.fy;
			// 	simulation.alpha(2).alphaTarget(0).restart();
			// }
			// align_node_dom();
			
			
			//RENAMING CODE
			let name = promptForName("Enter New Variable Name", obj.id, pdg.all_node_ids);
			if(!name) return;
			
			pdg.rename_node(obj.id, name);
		} else if(link) { // rename selected cpd
			
			
		} else { // nothing selected; create new variable here.
			setTimeout(function() {
				let name = promptForName("Enter A Variable Name", fresh_node_name(), pdg.all_node_ids);
				if(!name) return;
				
				newtgt = pdg.new_node(name, e.x, e.y);
				if(temp_link) {
					// todo: fold out this functionality, shared with click below.
					new_tgts = temp_link.tgts.slice(1);
					new_tgts.push(newtgt.id);
					pdg.new_link(temp_link.srcs, new_tgts, fresh_label(), [temp_link.x, temp_link.y]);
					temp_link = null;
				}
				pdg.tick();
			}, 10);
		}		
		// if(e.ctrlKey || e.metaKey) {
		// }
	});
	canvas.addEventListener("click", function(e) {
		// ADD NEW NODE
		// if(e.ctrlKey || e.metaKey) {
		if( temp_link ) {
			// let newtgt = pdg.pickN(e);
			// if(!newtgt && mode == 'draw') {
			// 	newtgt = new_node(fresh_node_name(), e.x, e.y);
			// }
			// if(newtgt) {
			// 	if(!e.shiftKey) {
			// 		new_tgts = temp_link.tgts.slice(1);
			// 		new_tgts.push(newtgt.id);
			// 		new_link(temp_link.srcs, new_tgts, fresh_label(), [temp_link.x, temp_link.y]);
			// 		temp_link = null;
			// 	}
			// 	else {
			// 		temp_link.tgts.push(newtgt.id);
			// 	}
			// }
			// console.log("IN CLICK W/ TEMP LINK")
			pdg.handle({
				type : "edge-stroke",
				temp_link : temp_link,
				endpt: {x : event.x, y : event.y},
				// source_link : event.subject.link
			});
			temp_link = null;

		} else if(action.type == 'move') {
			// mouse_end = vec2(lookup['<MOUSE>']);
			mouse_end = vec2(mouse);
			
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
			simulation.force("bipartite").links(mk_bipartite_links(linknodes));

			// for(let n of action.targets) {
			// 
			// }
			action = {};
			restyle_nodes();
			
		} else if(mode == 'move') { // selection in manipulate mode
			pdg.point_select(e, !e.shiftKey);
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
			pdg.tick();
		}
		else if (event.key == 'a') {
			pdg.select_all();
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
			src = pdg.selected_node_ids
			// src = nodes.filter( n => n.selected ).map( n => n.id );
			// lab = fresh_label();
			// temp_link = new_link(src, ['<MOUSE>'], "<TEMPORARY>");
			temp_link = linkobject(['<TEMPORARY>', [src, ["<MOUSE>"]]], undefined)
			if( src.length == 0) {
				temp_link.x = mouse.x
				temp_link.y = mouse.y
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
				pdg.delete_selection();
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
				mouse_start : vec2(mouse),
				targets: pdg.nodes.filter(n => n.selected).concat(pdg.linknodes.filter(ln => ln.link.selected)) 
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
		
		
		pdg.tick();
		// console.log(lover);
	});
	window.addEventListener("mousemove", function(e) {
		// mouse_pt = [e.x, e.y];
		// lookup["<MOUSE>"] = {x : e.x, y: e.y, w:0,h:0};
		// console.log("HI");
		// pdg.lookup["<MOUSE>"] = {x : e.x, y: e.y, w:0,h:0};
		
		mouse.x = e.x;
		mouse.y = e.y;
		
		if(temp_link) redraw();
		
		if(popped_up_link && !pdg.picksL(e, popped_up_link, 10)) {
			delete popped_up_link.lw;
			popped_up_link = null;
			pdg.tick();
		}
		
		if(popup_process) clearTimeout(popup_process);
	
		if( !popped_up_link) {
			popup_process = setTimeout(function() {
				let l = pdg.pickL(e, 10);
				popped_up_link = l;

				if(l) {
					l.lw = 5;
					pdg.tick();
				}
			}, 100);
		}

		// if ( mode == 'move'  && action) {
		if(action.type == 'move') {
			action.targets.forEach(n => {
				[n.x, n.y] = addv(n.old_pos, vec2(e), scale(action.mouse_start, -1)); 
			});
			pdg.restyle_nodes();
			// midpoint_aligning_force(1);
			pdg.tick();
			// TODO move selection, like ondrag below
		}
	})
});
