/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;
#include <cstdint>
#include <format>

export module slu.parse.adv.block;

import slu.ast.state;
import slu.lang.basic_state;
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
	StatList readGlobStatList(In& in)
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
	std::pair<StatList, lang::ModPathId> readStatList(
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