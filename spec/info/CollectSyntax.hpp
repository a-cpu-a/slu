/*
** Ai
*/
#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <regex>

namespace slu::spec {

    struct Rule
    {
        size_t lineDataIdx=SIZE_MAX;
        bool is_initial = false;
    };
    struct LineData
    {
        size_t aboveSpacing = 0;
        std::vector<std::string> lines;
    };

    std::string trim(const std::string& s) {
        auto begin = s.find_first_not_of(" \t\r\n");
        auto end = s.find_last_not_of(" \t\r\n");
        return (begin == std::string::npos) ? "" : s.substr(begin, end - begin + 1);
    }

    std::string extract_and_merge_ebnf_blocks(const std::string& directory = "in") {
        namespace fs = std::filesystem;
        std::vector<LineData> linesData;
        std::ostringstream output;
        std::unordered_map<std::string, Rule> rules;

        std::regex rule_start(R"(^\s*(\w+)\s*(::=|@::=)\s*(.*))");
        std::regex comment_line(R"(^\s*(--))");

        for (const auto& entry : fs::recursive_directory_iterator(directory))
        {
            if (!entry.is_regular_file() || entry.path().extension() == ".hpp")
                continue;

            std::ifstream file(entry.path());
            if (!file) continue;

            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            std::string_view sv(content);
            size_t pos = 0;
            const std::string start_token = "New Syntax {";

            while ((pos = sv.find(start_token, pos)) != std::string_view::npos)
            {
                size_t block_start = pos + start_token.size();
                size_t brace_count = 1;
                size_t i = block_start;

                while (i < sv.size() && brace_count > 0)
                {
                    if (sv[i] == '{') brace_count++;
                    else if (sv[i] == '}') brace_count--;
                    ++i;
                }

                if (brace_count != 0) break; // Unbalanced braces

                std::string block = std::string(sv.substr(block_start, i - block_start - 1));
                std::istringstream lines(block);
                std::string line;
                std::string current_rule;
                bool accumulating = false;
                size_t accumulIdx = 1;

                size_t linesAbove = 0;

                while (std::getline(lines, line))
                {
                    std::string trimmed = trim(line);

                    if (trimmed.empty()) 
                    {
                        linesAbove++;
                        continue;
                    }

                    std::smatch match;
                    if (std::regex_match(line, match, rule_start))
                    {
                        current_rule = match[1];
                        std::string op = match[2];
                        std::string rhs = match[3];


                        Rule& rule = rules[current_rule];
                        if (rule.lineDataIdx == SIZE_MAX)
                        {
                            rule.lineDataIdx = linesData.size();
                            linesData.emplace_back();
                        }
                        LineData& lData = linesData[rule.lineDataIdx];
                        if (op == "::=")
                        {
                            _ASSERT(!rule.is_initial);
                            rule.is_initial = true;

                            lData.aboveSpacing = linesAbove;
                            linesAbove = 0;

                            if (!lData.lines.empty())
                            {
                                lData.lines.front()+= rhs;
                            }
                            else
                            {
                                lData.lines.push_back("\t"+current_rule + " ::= " + rhs);
                            }
                        }
                        else
                        {
                            if (!lData.lines.empty())
                            {
                                lData.lines.push_back("\t\t"+rhs);
                            }
                            else
                            {
                                lData.lines.push_back("\t" + current_rule + " ::= ");
                                lData.lines.push_back("\t\t" + rhs);
                            }
                        }
                        accumulating = true;
                        accumulIdx = 1;
                    }
                    else if (std::regex_search(line, comment_line))
                    {
                        accumulating = false;
                        linesData.push_back({ .aboveSpacing = linesAbove, .lines = {line} });
                        linesAbove = 0;
                    }
                    else if (accumulating && (line.find('|') != std::string::npos || trim(line).front() == '|'))
                    {
                        auto& lines = linesData[rules[current_rule].lineDataIdx].lines;
                        lines.insert(lines.begin()+accumulIdx, line);
                        accumulIdx++;
                    }
                    else
                    {
                        accumulating = false;
                        linesData.push_back({ .aboveSpacing = linesAbove, .lines = {line} });
                        linesAbove = 0;
                    }
                }

                pos = i;
            }
        }

        // Write out comments and merged rules
        for (const LineData& line : linesData)
        {
            output << std::string(line.aboveSpacing,'\n');
            for (const auto& l : line.lines)
            {
                output << l << "\n";
            }
        }

        return output.str();
    }
}