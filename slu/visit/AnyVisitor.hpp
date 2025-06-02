/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <format>
#include <vector>

#include <slu/parser/State.hpp>
#include <slu/Settings.hpp>

namespace slu::visit
{
	/*
		(bool"shouldStop" pre_) post_ 
		sep_ -> commas ::'s etc
	*/
	template<class T>
	concept AnyVisitor = parse::AnyCfgable<T>;
}