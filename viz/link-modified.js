
function jiggle(random) {
    return (random()-0.5) *1E6;
}

function custom_link_force(blinks) {
    var id = l => l.indx,
        strength = (l) =>  1 / Math.min(count[l.source.index], count[l.target.index]),
        strengths,
        distance = (l) => [30,30],
        min_dists,
        max_dists,
        bnodes,
        count,
        bias,
        random,
        iterations = 3;
    var rect_pad_mode = true;
        
    const jiggl = () => (random()-0.5) *1E6;
    function vvec2(obj) {
        return [obj.x + obj.vx || jiggl(),
                obj.y + obj.vy || jiggl() ];
    }
    function w_h(obj) {
        return [obj.w, obj.h]
    }


    function force(alpha) {
        
        // console.log(distances);
        var dx,dy,source,target,bl, a,b, bi, max_dist, min_dist, r;

        for (var k = 0; k < iterations; ++k) {
            for(var i = 0; i < blinks.length; i++) {
                bl = blinks[i];
                source = bl.source, target = bl.target;
                
                if(rect_pad_mode) {
                    // let
                    a = sqshortened_end(vvec2(source), vvec2(target), w_h(target));
                    b = sqshortened_end(vvec2(target), vvec2(source), w_h(source));
                        
                    [dx,dy] = subv(a,b);
                } else {
                    // dx = target.x + target.vx - source.x - source.vx 
                    //     || jiggl();
                    // dy = target.y + target.vy - source.y - source.vy 
                    //     || jiggl();
                    [dx, dy] = subv(vvec2(target), vvec2(source));
                }
                r = Math.sqrt(dx * dx + dy * dy); 
                
                min_dist = bl.min_dist || min_dists[i];
                max_dist = bl.max_dist || max_dists[i];

                
                // console.log(min_dist,max_dist, dx,dy,r);
                r = (r - clamp(r, min_dist, max_dist)) / r * alpha * strengths[i];
                // console.log(min_dist,max_dist, dx,dy,r);

                dx *= r, dy *= r;
                
                // let b = 0.5; // TODO: make this count ratio as before.
                bi = bias[i]
                target.vx -= dx * bi;
                target.vy -= dy * bi;
                bi = 1 - bi;
                source.vx += dx * bi;
                source.vy += dy * bi;
            }
        
        }
    }
    function initialize() {
        // console.log("INITIALIZING")
        count = new Array(bnodes.length);
        let m = blinks.length,
            nodeById = new Map(bnodes.map((d, i) => [id(d, i, bnodes), d]));
        
        for (var i = 0; i < m; ++i) {
            let link = blinks[i];
            if (typeof link.source !== "object") link.source = nodeById.get(link.source);
            if (typeof link.target !== "object") link.target = nodeById.get(link.target);
            count[link.source.index] = (count[link.source.index] || 0) + 1;
            count[link.target.index] = (count[link.target.index] || 0) + 1;
        }
        for (i = 0, bias = new Array(m); i < m; ++i) {
            link = blinks[i], bias[i] = count[link.source.index] / (count[link.source.index] + count[link.target.index]);
        }
        
        max_dists = new Array(m); 
        min_dists = new Array(m);
        initializeDistance();
        strengths = new Array(m);
        initializeStrength();
    }
    force.initialize = function(_nodes, _random) {
        random = _random;
        bnodes = _nodes;
        initialize();
    };

    function initializeDistance() {
        if (!bnodes) return;

        for (var i = 0; i < blinks.length; ++i) {
            [min_dists[i], max_dists[i]] = distance(blinks[i], i, blinks);
        }
    }
    function initializeStrength() {
        if(!bnodes) return;
        for(var i = 0; i < blinks.length; ++i)
            strengths[i] = +strength(blinks[i],i,blinks)
    }
    
    force.links = function(_) {
        return arguments.length ? (blinks = _, initialize(), force) : links;
    };

    force.id = function(_) {
        return arguments.length ? (id = _, force) : id;
    };
    force.iterations = function(_) {
        return arguments.length ? (iterations = +_, force) : iterations;
    };
    force.distance = function(_) {
        return arguments.length ?
            (distance = typeof _ === "function" ? _ : () => [_,_] , initializeDistance(), force)
            : distance;
    };
    force.strength = function(_) {
        return arguments.length ?
            (strength = typeof _ === "function" ? _ : () => _ , initializeStrength(), force)
            : strength;
    };
    force.rect_pad_mode = function( _ ) {
        return arguments.length ? (rect_pad_mode = _, force): rect_pad_mode;
    }
    return force
}
