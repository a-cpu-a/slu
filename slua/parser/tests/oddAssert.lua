
local co = coroutine.wrap(f)
assert(co() == 's')
assert(co() == 'g')
assert(co() == 'g')
assert(co() == 0)

assert(X[lim] == lim - 1 and X[lim + 1] == lim)

-- errors in accesses larger than K (in RK)
getmetatable(env).__index = function () end