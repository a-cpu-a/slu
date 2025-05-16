/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/Settings.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/State.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/paint/SemOutputStream.hpp>
#include <slua/paint/Paint.hpp>

namespace slua::paint
{
	/*
	ps, add something like this:
	```css
	.slua-code-box {
		tab-size: 3;
		white-space-collapse: preserve;
		background:#151515;
		padding:8px;
		border-radius:8px;
		margin: 4px 0px;
		display: block;
	}
	```
	*/
	inline std::pair<std::string,uint32_t> 
		getCssFor(const AnySemOutput auto& se, const bool makeCssStr) 
	{
		std::unordered_map<uint32_t,size_t> colors;

		for (const auto& chList : se.out)
		{
			for (const uint32_t chCol : chList)
			{
				colors[chCol & 0xFFFFFF]++;
			}
		}

		uint32_t mostCommonCol = UINT32_MAX;
		size_t mostCommonColCount = 0;

		std::string res;
		for (const auto [col,count]: colors)
		{
			if(mostCommonColCount < count)
			{
				mostCommonColCount = count;
				mostCommonCol = col;
			}

			res += ".C";
			for (size_t i = 0; i < 6; i++)
			{
				const uint8_t nibble = (col >> (20 - i * 4)) & 0xF;
				if (nibble < 10)
					res += '0' + nibble;
				else
					res += 'A' + (nibble - 10);
			}
			res += "{color:#";
			for (size_t i = 0; i < 6; i++)
			{
				const uint8_t nibble = (col >> (20 - i * 4)) & 0xF;
				if (nibble < 10)
					res += '0' + nibble;
				else
					res += 'A' + (nibble - 10);
			}
			res += '}';
		}
		return { res, mostCommonCol};
	}
	/*
	Make sure to reset in first: `in.reset();`
	*/
	inline std::string toHtml(AnySemOutput auto& se,const bool includeStyle) {
		std::string res;

		auto [cssStr, mostCommonCol] = getCssFor(se, includeStyle);
		if (includeStyle)
		{
			res += "<style>" + cssStr + "</style>";
		}

		if (mostCommonCol == UINT32_MAX)
		{// Its empty
			res += "<code class=slua-code-box></code>";
			return res;
		}

		res += "<code class=\"slua-code-box C";
		for (size_t i = 0; i < 6; i++)
		{
			const uint8_t nibble = (mostCommonCol >> (20 - i * 4)) & 0xF;
			if (nibble < 10)
				res += '0' + nibble;
			else
				res += 'A' + (nibble - 10);
		}
		res += "\">";
		uint32_t prevCol = 0xFFFFFF;
		bool closeSpan = false;

		parse::ParseNewlineState nlState = parse::ParseNewlineState::NONE;
		while (se.in)
		{
			const parse::Position loc = se.in.getLoc();
			const char ch = se.in.get();

			//Already skipping
			if (parse::manageNewlineState<false>(ch, nlState, se.in))
			{
				res += "<br>";
				continue;
			}

			if (parse::isSpaceChar(ch))
			{
				res += ch;
				continue;//Dont explicitly paint it!
			}

			//TODO: fix space skip code, so it doesnt do this:
			uint32_t col = se.commentPair();

			if(loc.line<se.out.size())
			{
				const auto& lineCols = se.out[loc.line - 1];
				if(loc.index < lineCols.size())
					col = lineCols[loc.index] & 0xFFFFFF;
			}

			if (prevCol != col)
			{
				prevCol = col;

				if(closeSpan)
					res += "</span>";
				if (col == mostCommonCol)
					closeSpan = false;
				else
				{
					closeSpan = true;
					res += "<span class=C";
					for (size_t i = 0; i < 6; i++)
					{
						const uint8_t nibble = (col >> (20 - i * 4)) & 0xF;
						if (nibble < 10)
							res += '0' + nibble;
						else
							res += 'A' + (nibble - 10);
					}
					res += ">";
				}
			}

			if (ch == '\'')
			{
				res += "&#39;";
				continue;
			}
			if (ch == '\"')
			{
				res += "&#34;";
				continue;
			}
			if (ch == '>')
			{
				res += "&gt;";
				continue;
			}
			if (ch == '<')
			{
				res += "&lt;";
				continue;
			}
			if (ch == '&')
			{
				res += "&amp;";
				continue;
			}
			res += ch;
		}
		if (closeSpan)
			res += "</span></code>";
		else
			res += "</code>";
		return res;
	}
}