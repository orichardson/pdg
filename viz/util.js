


//## Create a new subobject with the subset of properties
function cloneAndPluck(sourceObject, keys) {
    let newObject = {};
    keys.forEach((key) => { newObject[key] = sourceObject[key]; });
    return newObject;
}
function promptForName(prompt, defaultname, taken) {
	let name = window.prompt(prompt, defaultname);
	if(name) {
		if(taken.includes(name)) {
			window.alert(`name ${name} already taken.`);
		}
		else return name;
	}
}

//## DOWNLOAD
function download_JSON(obj, fname) {
	let str = JSON.stringify(obj);
	const file = new File([str], fname+'.json', {type: 'text/json'});
	
	const link = document.createElement('a')
     const url = URL.createObjectURL(file)

     link.href = url
     link.download = file.name
     document.body.appendChild(link)
     link.click()

     document.body.removeChild(link)
     window.URL.revokeObjectURL(url)
}

// ## BASIC VECTOR OPS
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
// ## GEOMETRY
function corners2xywh(start, end) {
	xmin = Math.min(start[0], end[0]);
	xmax = Math.max(start[0], end[0]);
	ymin = Math.min(start[1], end[1]);
	ymax = Math.max(start[1], end[1]);
	// return [xmin,xmax,ymin,ymax];
	return [xmin, ymin, xmax-xmin, ymax-ymin];
}
function pt_in_rect(pt, rectobj) {
    return (Math.abs(pt[0] - rectobj.x) < rectobj.w/2) &&
           (Math.abs(pt[1] - rectobj.y) < rectobj.h/2);
}

// ## More Complex Graphics Computations
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


// ## FORCE MODIFICATION TO SUBSET
function filtered_force(force, nodeFilter) {
    let init = force.initialize;
    force.initialize = function(_nodes, _random) {
        init(_nodes.filter(nodeFilter), _random);
    }
    return force;
}
