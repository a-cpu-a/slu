

//This file is used to test compilation

//Test macro, dont use, doesnt improve performace. (actually hurts it lol)
//#define Slua_NoConcepts
#include <slua/Include.hpp>
#include <slua/ErrorType.hpp>
#include <slua/Context.hpp>
#include <slua/MetaTableUtils.hpp>
#include <slua/WrapFunc.hpp>
#include <slua/Utils.hpp>

#include <slua/types/Converter.hpp>
#include <slua/types/ReadWrite.hpp>
#include <slua/types/TypeUtils.hpp>
#include <slua/types/UserData.hpp>
#include <slua/types/complex/AnyRef.hpp>
#include <slua/types/complex/OptFunc.hpp>
#include <slua/types/complex/Vector.hpp>
#include <slua/types/complex/Function.hpp>
#include <slua/types/basic/StackItem.hpp>
#include <slua/types/basic/Optional.hpp>
#include <slua/types/basic/RegistryRef.hpp>

#include <slua/parser/Parse.hpp>
#include <slua/parser/VecInput.hpp>
#include <slua/generator/Gen.hpp>
#include <slua/MetaTableUtils.hpp>


void _test()
{

	slua::parse::VecInput in;
	slua::parse::parseFile(in);

	slua::parse::VecInput in2{ slua::parse::sluaSyn
		| slua::parse::noIntOverflow
		| slua::parse::spacedFuncCallStrForm 
	};
	slua::parse::parseFile(in2);

	slua::parse::Output out;
	slua::parse::genFile(out, {});
}