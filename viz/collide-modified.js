// import {quadtree} from "d3-quadtree";
// import constant from "./constant.js";
// import jiggle from "./jiggle.js";

// copied from other file. Only need one. Oops.
// function jiggle(random) {
//     return (random()-0.5) *1E6;
// }

function custom_collide_force() {
	var nodes,
		random,
		strength = 1,
		iterations = 1;

	// if (typeof radius !== "function")
	// 	radius = () => (radius == null ? 1 : +radius);

	function force() {
		var i, n = nodes.length,
			tree,
			node,
			xi,yi, 
			wi,hi;

		for (var k = 0; k < iterations; ++k) {
			tree = d3.quadtree(nodes, d => d.x + d.vx, d => d.y + d.vy).visitAfter(prepare);
			for (i = 0; i < n; ++i) {
				node = nodes[i];
				// ri = radii[node.index], ri2 = ri * ri;
				wi = node.w;
				hi = node.h;
				xi = node.x + node.vx;
				yi = node.y + node.vy;
				tree.visit(apply);
			}
		}

		function apply(quad, x0, y0, x1, y1) {
			var data = quad.data;
			let rx = (wi+quad.w)/2, ry = (hi+quad.h)/2;
			if (data) {
				// why is this so impiortant?
				// if (data.index > node.index) {
				if (data.index < node.index) {
				// if(true){
					// corner point
					let [datx,daty] = [data.x+data.vx, data.y+data.vy]
					let [ex,ey] = sqshortened_end([xi,yi], [datx,daty], [data.w, data.h], 0) 
					// let edge2 = sqshortened_end([xi,yi], [data.x+data.vx, data.y+data.vy], [data.w, data.h]) 

					// var x = xi - data.x - data.vx,
					//     y = yi - data.y - data.vy;
					// l = x * x + y * y;
					
					// if edge point is within the rectangle defined by node...
					
					// let [ex2,ey2] = sqshortened_end([ex,ey], [xi,yi], [wi,hi])
					let [ex2,ey2] = sqshortened_end([datx,daty], [xi,yi], [wi,hi], 0)
					
					
					if(i == 0){
						[node.ex, node.ey] = [ex,ey];
						[node.ex2, node.ey2] = [ex2,ey2];
					}
					
										
					// if( (ex < xi + wi/2 && ex > xi - wi/2 
					// 		&& ey < yi + wi/2 && ey > yi - wi/2)
					// 	|| (ex2 < datx + data.w/2 && ex2 > datx -data.w/2 
					// 		&& ey2 < daty + data.h/2 && ey2 > daty -data.h/2)) {
					// if( (ex2 < xi + wi/2 && ex2 > xi - wi/2 
					// 		&& ey2 < yi + wi/2 && ey2 > yi - wi/2)
					// 	|| (ex < datx + data.w/2 && ex > datx - data.w/2 
					// 		&& ey < daty + data.h/2 && ey > daty - data.h/2)) {
					if( (ex < xi + wi/2 && ex > xi - wi/2 
					 		&& ey < yi + wi/2 && ey > yi - wi/2)
					 	|| (ex2 < datx + data.w/2 && ex2 > datx -data.w/2 
					 		&& ey2 < daty + data.h/2 && ey2 > daty -data.h/2)) {

						// if (x === 0) x = jiggle(random), l += x * x;
						// if (y === 0) y = jiggle(random), l += y * y;
						//
						// l = (r - (l = Math.sqrt(l))) / l * strength;
						// let dx = ex-ex2, dy = ey - ey2;
						// if (dx === 0) dx = jiggle(random);
						// if (dy === 0) dy = jiggle(random);
						
						let dx = datx - xi - rx*sgn(datx - xi),
							dy = daty - yi - ry*sgn(daty - yi);
						
						
						if(dx * dx > dy*dy) dx = 0;
						else dy = 0;						
						
						// let factor = Math.sqrt(dx*dx + dy*dy) / (Math.max(datx,daty)) * strength;
						// let factorx = (rx - dx) / dx *strength,
						// 	factory = (ry - dy) / dy * strength;
						// let factorx = factory = 0.1;
						let factorx = factory = strength;
						
						dx *= factorx;
						dy *= factory;
						
						let portion = data.w*data.h / (data.w*data.h + wi*hi);
						// let portion = 0.5;
						
						node.vx += dx * portion;
						node.vy += dy * portion;
						data.vx -= dx * (1-portion);
						data.vy -= dy * (1-portion);
					}
				}
				return;
			}
			// let rx = (wi+quad.w)/2, ry = (hi+quad.h)/2;
			return x0 > xi + rx || x1 < xi - rx || y0 > yi + ry || y1 < yi - ry;
			// return false;
		}
	}

	function prepare(quad) {
		// if (quad.data) // if leaf node, set radius. 
		// 	return quad.r = radii[quad.data.index];
		// if not a leaf, set radius to largest of children.
		// for (var i = quad.r = 0; i < 4; ++i) {
		// 	if (quad[i] && quad[i].r > quad.r) {
		// 		quad.r = quad[i].r;
		// 	}
		// }
			
		if(quad.data) {
			quad.w = quad.data.w;
			quad.h = quad.data.h;
		}
		else {
			for (var i = quad.r = 0; i < 4; ++i) {
				if (quad[i] && quad[i].w > quad.w) {
					quad.w = quad[i].w;
				}
				if(quad[i] && quad[i].h > quad.h) {
					quad.h = quad[i].h;
				}
			}
		}
	}

	force.initialize = function(_nodes, _random) {
		nodes = _nodes;
		random = _random;
	};

	force.iterations = function(_) {
		return arguments.length ? (iterations = +_, force) : iterations;
	};

	force.strength = function(_) {
		if(arguments.length) {
			strength = +_; 
			return force;
		}
		return strength
	};
	
	return force;
}
