function u8(str)
  return "utf "..str
end
function getFirst(arr)
  return arr[1]
end

local y = u8"<-- lol"

local uw = getFirst{"elem 1"}

function s(s)return s end

function string.aa(str,tbl) return str..tostring(tbl); end
function string.ziq(str,addStr) return str..addStr; end
function string.u8(bStr,str)
  return bStr.." "..str
end

print(s"aa":ziq"over":u8"ppl ":aa{1,2,3,[35]=3,nam="e",3,[34]=3,name="e"})