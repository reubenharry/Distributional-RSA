


var world_prior = function(){
  Infer({method:'enumerate',model: function(){
    var vicious = flip(0.75)
    var animal = flip(0.1)
    return {vicious:vicious,animal:animal}
  }})}

var utterances = ["shark","silence"]



var sem = function(u,w){
  return u=="shark"?w["vicious"]&&w["animal"]:true
}

var l0 = function(utterance){
  Infer({method:'enumerate',model: function(){
    var world = sample(world_prior())
    condition(sem(utterance,world))
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
    var w = sample(world_prior())
    var q = uniformDraw(["vicious","animal"])
    factor(s1(w,q).score(utterance))
    return [w,q]
    
  }})
}

var l1Prior = function(){
  Infer({model: function(){
    var w = sample(world_prior())
    var q = uniformDraw(["vicious","animal"])
    return [w,q]
    
  }})
}

viz.table(l0('shark'))
viz.table(l0('silence'))
// viz.table(s1(worlds[0],"vicious"))
viz.table(l1('shark'))
viz.table(l1('silence'))
viz.table(l1Prior())
// viz.table(l1('fish'))
