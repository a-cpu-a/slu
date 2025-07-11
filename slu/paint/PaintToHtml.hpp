﻿/*
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

#include <slu/Settings.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/State.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/paint/SemOutputStream.hpp>
#include <slu/paint/Paint.hpp>

namespace slu::paint
{
	/*
	ps, add something like this:
	```css
	.slu-code-box {
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
		uint32_t prevCol = 0xFF00FF;
		for (const auto& chList : se.out)
		{
			for (uint32_t chCol : chList)
			{
				chCol &= 0xFFFFFF;

				if (prevCol == chCol)
					continue;
				prevCol = chCol;
				colors[chCol]++;
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
	//SIZE_MAX when not found
	inline size_t locateColStackIndex(const std::vector<uint32_t>& colStack,const uint32_t col)
	{
		size_t i = colStack.size()-1;
		for (const uint32_t c : std::ranges::reverse_view{ colStack })
		{
			if(c == col)
				return i;
			i--;
		}
		return SIZE_MAX;
	}
	/*
	Make sure to reset in first: `in.reset();`

	DEFLATE seems to like high nestLimit
	anything more complex seems to like a nestLimit of 1
	no compression also likes high nestLimit
	*/
	inline std::string toHtml(AnySemOutput auto& se,const bool includeStyle,const size_t nestLimit=32) {
		std::string res;

		auto [cssStr, mostCommonCol] = getCssFor(se, includeStyle);
		if (includeStyle)
		{
			res += "<style>" + cssStr + "</style>";
		}

		if (mostCommonCol == UINT32_MAX)
		{// Its empty
			res += "<code class=slu-code-box></code>";
			return res;
		}

		res += "<code class=\"slu-code-box C";
		for (size_t i = 0; i < 6; i++)
		{
			const uint8_t nibble = (mostCommonCol >> (20 - i * 4)) & 0xF;
			if (nibble < 10)
				res += '0' + nibble;
			else
				res += 'A' + (nibble - 10);
		}
		res += "\">";
		std::vector<uint32_t> colStack;
		colStack.push_back(mostCommonCol);

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
				if(ch!='\r')
					res += ch;
				continue;//Dont explicitly paint it!
			}

			const uint32_t col = se.out[loc.line - 1][loc.index] & 0xFFFFFF;

			if (colStack.back() != col)
			{
				// Go down the stack, or go up
				size_t idx = locateColStackIndex(colStack, col);
				if (idx == SIZE_MAX)
				{
					// Check nest limit, if we are too deep, then we need to pop the stack
					if (colStack.size() > nestLimit)
					{
						colStack.pop_back();
						res.append("</span>",7);
					}
					colStack.push_back(col);
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
				else
				{
					//We are going up the stack, so we need to pop the stack until we reach the right color
					const size_t popCount = colStack.size() - idx - 1;
					const size_t start = res.size();
					res.resize(start + popCount * 7);
					char* d = res.data();
					for (size_t i = 0; i < popCount; i++)
					{
						const size_t j = start + i * 7;
						d[j+0]='<';
						d[j+1]='/';
						d[j+2]='s';
						d[j+3]='p';
						d[j+4]='a';
						d[j+5]='n';
						d[j+6]='>';
					}
					colStack.resize(idx + 1);//remove the popped colors
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
		for (size_t i = 0; i < colStack.size()-1; i++)
			res += "</span>";
		res += "</code>";
		return res;
	}
}