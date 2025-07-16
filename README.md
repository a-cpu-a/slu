# <img alt="Slu Lang logo - hollow star with a cresent going through the middle" src="/spec/info/Logo_white.png" width="120"> <img alt="Slu Lang" src="/spec/info/LogoText.svg" width="150"> 
 
Wip slu compiler/parser/linter currently written in C++ 20. 

The goal is to make a safe and fast language, that is easy to transpile into other languages, ... 

## Features

Builtin support for transpilation (todo: maybe make a mlir dialect for each?) (lua, hopefully: js, jvm bytecode, ...)  
Types as values (no generics, just functions: `Vec(u8)`)  
Safety checking  
Borrow checking (hopefuly easy to understand with lifetimes being just variable names: `&/var1/var2 T`)  
Structural and nominal types  
Trait, impl pattern  
Ranged integers `const u8 = 0..0xFF`, currently out of range stuff is (planned to be) a
warning + handled at runtime, unless the ranges dont overlap, then it would be a error  
Builtin support for a result type `throw MyErr{"oh no"}`  
Compile-time code execution (todo: sandbox it to make it safe & deterministic: mlir->wasm?)  

[Spec is located here](/spec/)
