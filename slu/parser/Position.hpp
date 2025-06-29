/*
** See Copyright Notice inside Include.hpp
*/
#pragma once


namespace slu::parse
{
	struct Position
	{
		size_t line;
		size_t index;//Oops 0 based! TODO: fix that!
	};
}