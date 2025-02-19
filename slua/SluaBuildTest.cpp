

//This file is used to test compilation


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

#include <slua/parser/Parse.hpp>
#include <slua/generator/Gen.hpp>
#include <slua/MetaTableUtils.hpp>


void _test()
{
	sluaParse::Input in;
	sluaParse::parseFile(in);
}