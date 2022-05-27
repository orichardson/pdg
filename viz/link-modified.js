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
        iterations = 5;
        

    function force(alpha) {
        // console.log(distances);
        for (var k = 0; k < iterations; ++k) {
            for(var i = 0; i < blinks.length; i++) {
                let bl = blinks[i];
                let source = bl.source, target = bl.target;
                let dx = target.x + target.vx - source.x - source.vx 
                    || Math.random() * 10 - 5;
                    //|| jiggle(random);
                let dy = target.y + target.vy - source.y - source.vy 
                    //|| jiggle(random);
                    || Math.random() * 10 - 5;
                let r = Math.sqrt(dx * dx + dy * dy);
                let min_dist = bl.min_dist || min_dists[i];
                let max_dist = bl.max_dist || max_dists[i];
                
                // console.log(min_dist,max_dist, dx,dy,r);
                r = (r - clamp(r, min_dist, max_dist)) / r * alpha * 1;
                
                // console.log(min_dist,max_dist, dx,dy,r);

                dx *= r, dy *= r;
                
                // let b = 0.5; // TODO: make this count ratio as before.
                let b = bias[i]
                target.vx -= dx * b;
                target.vy -= dy * b;
                b = 1 - b;
                source.vx += dx * b;
                source.vy += dy * b;
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
        for(var i = 0; i < m; ++i) strengths[i] = +strength(blinks[i],i,blinks)
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
    return force
}
