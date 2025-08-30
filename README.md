# <img alt="Slu Lang logo - hollow star with a cresent going through the middle" src="/spec/info/Logo_white.png" width="120"> <img alt="Slu Lang" src="/spec/info/LogoText.svg" width="150"> 
 
Wip slu compiler/parser/linter currently written in C++ 20. 

The goal is to make a safe and fast language, that is easy to transpile into other languages, ... 

## Features

Builtin support for transpilation (todo: maybe make a mlir dialect for each?) (js, hopefully: lua, jvm bytecode, ...)  
Types as values (no generics, just functions: `Vec(u8)`)  
Trait/impl based type extensions.  
Structural and nominal types  
Ranged integers `const u8 = 0..0xFF as type`, currently out of range stuff is (planned to be) either a
warning + handled at runtime, or a compilation error... unless the ranges dont overlap, then it would obviously be a error.  
Borrow checking (hopefuly easy to understand with lifetimes being just variable names: `&/var1/var2 T`)  
Safety checking  
Builtin support for a result type `throw MyErr{"oh no"}`  
Compile-time code execution (todo: sandbox it to make it safe & deterministic: mlir->wasm?)  
Less global state by default, requiring a function call to obtain access to time, files, logging, etc. (they are trait based, allowing you to wrap and intercept uses)  
impl types are checked before any monomorphization, automatically support dyn types where possible.  

[Spec is located here](/spec/)  

```
fn printHelloWorld(l = &mut impl std::Log) {
	l.info("Hello world!", {}); -- The {} is for formatting arugments (no arguments in this example).
}
fn main() 
{
	-- Since logging is not always possible, you have to explicitly get a logger.
	-- In this case we want the log message to always be shown, so we panic if there is no logger.
	-- If logging is not important, you can use `std::getDefaultLogger()` which returns an option.
	-- or `std::getDefaultLoggerOrDummy()` which may return a dummy implementation that voids your logs.
	let mut impl std::Log l = std::getDefaultLoggerOrPanic();
	printHelloWorld(&mut l);
}
```