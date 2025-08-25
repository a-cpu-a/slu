module;
/*
** See Copyright Notice inside Include.hpp
*/
export module slu.parse.pos;
namespace slu::parse
{
	export struct Position
	{
		size_t line;
		size_t index;//Oops 0 based! TODO: fix that!
	};
}