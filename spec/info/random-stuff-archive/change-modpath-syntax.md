
slu:std:Vec(slu:std:u8)
slu::std::Vec(slu::std::u8)
slu.:std.:Vec(slu.:std.:u8)


# Why change
Currently `:` is reserved for "liquid" types `T : |v| v!=0`
That is too short for something kinda complex
`::` would work very nicely too `T :: |v| v!=0`, looks nicer actually.
`::` would have no issues in pattern types `{x :: u8 :: |v| v!=0} :: Ok =>`



# Cant do this one, as it would conflict with any module reflection efforts
`slu.std.Vec(slu.std.u8)`
Like, module values, and methods on them:  
```
--let Module thisMod = mod -- Still not implemented syntax to replace `self::something`
let Module m = slu::std;
let type T = m.getItemByName("u8")
```
Better to keep it reserved for the future.
## Notes 
Would imply that module values have fields & methods auto implemented, meaning they are kinda like structs with some constant valued fields.  
"first class modules", which would hurt recursive module use


# Why keep it

`::` offers more space, maybe making it easier to read.
Using `:` for "liquid" types makes it easier to use them.

