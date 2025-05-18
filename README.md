# <img alt="Slu Lang logo - hollow star with a cresent going through the middle" src="/spec/info/Logo_white.png" width="120"> <img alt="Slu Lang" src="/spec/info/LogoText.svg" width="150"> 
 
Wip slu compiler/parser/linter currently written in C++ 20. 

The goal is to make a safe and fast language, that is easy to transpile into other languages and bytecodes, but also ir's like llvm ir. 

Types as values (no generics, just functions: `Vec(u8)`)  
Safety checking  
Borrow checking (hopefuly easy to understand with lfetimes being just variable names: `&/var1/var2 T`)  
Structural and nominal types  
Ranged integers `const u8 = 0...0xFF`, currently out of range stuff is (planned to be) handled at runtime, unless they dont overlap  
Builtin result type `throw MyErr{"oh no"}`  
Compile-time code execution (Must be safe to execute. might output ur personal files tho... maybe?)  

[Spec is located here](/spec/)
