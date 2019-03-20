
var world_priors = [0.15,0.05,0.35,0.45]
var worlds = [
    {vicious: true, aquatic: true}, 
    {vicious: true, aquatic: false}, 
    {vicious: false, aquatic: true},
    {vicious: false, aquatic: false},
    ]

var utterances = ["shark","fish"]

var sem = function(u,w){
  return u=="shark"?w["vicious"]&&w["aquatic"]:w["aquatic"]
}

var l0 = function(utterance){
  Infer({method:'enumerate',model: function(){
    var world = categorical({vs: worlds, ps: world_priors })
    var utility = sem(utterance,world)
    factor(utility)
    return world    
  }})}

var s1 = function(w,q){
  Infer({model: function(){
    var u = uniformDraw(utterances)
    var l0_belief = sample(l0(u))
    condition(w[q]==l0_belief[q])
    return u
  }})}

var l1 = function(utterance){
  Infer({model: function(){
    var w = categorical({vs: worlds, ps: world_priors })
    var q = uniformDraw(["vicious","aquatic"])
    factor(s1(w,q).score(utterance))
    return [w]
    
  }})
}

viz.table(l0('shark'))
viz.table(l1('shark'))
viz.table(l1('fish'))
