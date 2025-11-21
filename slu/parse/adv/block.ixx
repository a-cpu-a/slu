module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
#include <format>

export module slu.parse.adv.block;

import slu.ast.state;
import slu.parse.input;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	//startCh == in.peek() !!!
	inline bool isBasicBlockEnding(AnyInput auto& in, const char startCh)
	{
		if (startCh == '}')
			return true;
		if (startCh == 'u')
		{
			if (checkTextToken(in, "until"))
				return true;
		} else if (startCh == 'e')
		{
			if (checkTextToken(in, "else"))
				return true;
		}
		return false;
	}

	export RetType readRetStart(AnyInput auto& in)
	{
		if (checkReadTextToken<false>(in, "return"))
			return RetType::RETURN;
		else if (checkReadTextToken<false>(in, "break"))
			return RetType::BREAK;
		else if (checkReadTextToken<false>(in, "continue"))
			return RetType::CONTINUE;
		return RetType::NONE;
	}

	export template<bool isLoop, AnyInput In>
	bool readReturnAfterStart(
	    In& in, const bool allowVarArg, const RetType retTy)
	{
		if constexpr (!isLoop)
		{
			if (retTy == RetType::BREAK //TODO: allow in some more contexts.
			    || retTy == RetType::CONTINUE)
			{
				in.handleError(std::format("Break used outside of loop"
				                           "{}",
				    errorLocStr(in)));
			}
		}
		if (retTy != RetType::NONE)
		{
			in.genData.scopeReturn(retTy);

			skipSpace(in);

			if (!in)
				return true;

			const char ch1 = in.peek();

			if (ch1 == ';')
				in.skip(); //thats it
			else if (!isBasicBlockEnding(in, ch1))
			{
				in.genData.scopeReturn(readExprList(in, allowVarArg));

				readOptToken(in, ";");
			}
			return true;
		}
		return false;
	}
	export template<bool isLoop>
	bool readReturn(AnyInput auto& in, const bool allowVarArg)
	{
		skipSpace(in);
		if (!in)
			return false;
		return readReturnAfterStart<isLoop>(in, allowVarArg, readRetStart(in));
	}

	export template<bool pushAnonScope, AnyInput In>
	StatList<In> readGlobStatList(In& in)
	{
		if constexpr (pushAnonScope)
			in.genData.pushAnonScope(in.getLoc());
		skipSpace(in);
		while (true)
		{
			if (!in) //File ended, so stat-list ended too
				break;

			if (in.peek() == '}')
				break;

			readStat<false>(in, false);
			skipSpace(in);
		}
		return in.genData.popScope(in.getLoc()).statList;
	}
	//second -> nextSynName, only for isGlobal=true
	//No start '{' check, also doesnt skip '}'!
	export template<bool isLoop, AnyInput In>
	std::pair<StatList<In>, lang::ModPathId> readStatList(
	    In& in, const bool allowVarArg, const bool isGlobal)
	{
		skipSpace(in);
		in.genData.pushUnScope(in.getLoc(), isGlobal);

		while (true)
		{
			if (!in) //File ended, so stat-list ended too
				break;

			if (in.peek() == '}')
				break;

			readStat<isLoop>(in, allowVarArg);
			skipSpace(in);
		}
		Block<In> bl = in.genData.popUnScope(in.getLoc());
		return {std::move(bl.statList), bl.mp};
	}

	//if not pushScope, then you will need to push it yourself!
	export template<bool isLoop, AnyInput In>
	Block<In> readBlock(In& in, const bool allowVarArg, const bool pushScope)
	{
		/*
		    block ::= {stat} [retstat]
		    retstat ::= return [explist] [‘;’]
		*/

		skipSpace(in);

		if (pushScope)
			in.genData.pushAnonScope(in.getLoc());

		while (true)
		{

			if (!in) //File ended, so block ended too
				break;

			const char ch = in.peek();

			bool mayReturn = ch == 'r' || ch == 'b' || ch == 'c';

			if (mayReturn)
			{
				if (readReturn<isLoop>(in, allowVarArg))
					break; // no more loop
			} else if (isBasicBlockEnding(in, ch))
				break; // no more loop

			// Not some end / return keyword, must be a statement

			readStat<isLoop>(in, allowVarArg);

			skipSpace(in);
		}
		return in.genData.popScope(in.getLoc());
	}

	export template<bool isLoop, AnyInput In>
	Block<In> readBlockNoStartCheck(
	    In& in, const bool allowVarArg, const bool pushScope)
	{
		Block<In> bl = readBlock<isLoop>(in, allowVarArg, pushScope);
		requireToken(in, "}");

		return bl;
	}
} //namespace slu::parse