import asyncio
import os
import aiohttp
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime
from exa_py import Exa


class SearchResult(BaseModel):
    query: str
    url: str
    title: str
    content: str
    score: float=Field(ge=0.0, le=1.0, description='Relevance score from 0-1')

class SourceSummary(BaseModel):
    source: str
    url: str
    key_points: List[str] = Field(min_items=1, max_items=8, description='Main Insight')
    confidence: float = Field(ge=0.0, le=1.0, description='Summary Confidence')
    content_quality: str = Field(description='high/medium/low quality assessment')


class ResearchCritique(BaseModel):
    overall_quality: float=Field(ge=0.0, le=1.0, description='Overall research quality')
    source_credibility: float = Field(ge=0.0, le=1.0, description='Source credibility score')
    coverage_gaps: List[str] = Field(description='Missing Information areas')
    potential_biases: List[str] = Field(description='Identified biases or limitations')
    consistency_issues: List[str] = Field(description='Contradictions or inconsistencies')
    recommendations: List[str] = Field(description='How to improve this research')

class ResearchReport(BaseModel):
    topic: str
    timestamp: datetime
    executive_summary: str
    main_findings: List[str]
    implications: List[str]
    knowledge_gaps: List[str]
    conclusions: List[str]
    sources_analyzed: int
    research_quality_score: float=Field(ge=0.0, le=10.0)

class CompleteResearch(BaseModel):
    topic: str
    search_results: List[SearchResult]
    summaries: List[SourceSummary]
    critique: ResearchCritique
    final_report: ResearchReport
    timestamp: datetime


class LlamaCppClient:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    async def generate(self, prompt:str, max_tokens: int=512, temperature:float = 0.7):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.base_url}/completion",
                        json={
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "stream": False
                            },
                        timeout=aiohttp.ClientTimeout(total=60)
                        )as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result['content'].strip()
                    else:
                        raise Exception(f"HTTP {resp.status}: {await resp.text()}")
        except Exception as e:
            print(f"Error with {self.model_name}: {e}")
            return f"Error: {str(e)}"
    
    async def generate_structured(self, prompt:str, response_model: BaseModel, max_retries: int=3):
        for attempt in range(max_retries):
            try:
                structured_prompt = f'''{prompt}

                IMPORTANT: Respond with valid JSON only, no additional text. Use this exact format:
                {response_model.model_json_schema()}

                JSON Response:'''

                response = await self.generate(structured_prompt, max_tokens=800, temperature=0.3)
                json_str = self._extract_json(response)

                return response_model.model_validate_json(json_str)
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"⚠️  Attempt {attempt + 1} failed for {self.model_name}: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to get valid structure output after {max_retries} attempts")
        return None

    def _extract_json(self, text:str) -> str:

        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError('NO JSON found in response')

        brace_count = 0
        end_idx = None
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if char=='{':
                brace_count += 1
            elif char =='}':
                brace_count -= 1
                if brace_count ==0:
                    end_idx = i+1
                    break

        if end_idx is None:
            raise ValueError("Unbalanced braces in text")

        return text[start_idx:end_idx]


class WebSearchAgent:
    def __init__(self, llama_client:LlamaCppClient, exa_api_key: str):
        self.client = llama_client
        self.exa = Exa(exa_api_key)

    async def search_web(self, topic: str) -> List[SearchResult]:
        query_prompt = f'''Generate 3 focused search queries for reasoning "{topic}". Make them specific and targeted for finding recent, authoritative information.

Topic: {topic}'''

        query_text = await self.client.generate(query_prompt, max_tokens=200, temperature=0.3)

        queries = [line.strip('- ').strip() for line in query_text.split('\n')
                if line.strip() and not line.startswith('Topic:')][:3]

        print(f'🔍 Generated queries: {queries}')

        all_results = []

        for query in queries:
            try:
                print(f"🌐 Searching Exa for: {query}")

                search_response = self.exa.search_and_contents(
                        query=query,
                        type='neural',
                        num_results=2,
                        text=True,
                        highlights=True)
                for result in search_response.results:
                    content = result.text[:2000] if result.text else ""
                    if result.highlights:
                        content = result.highlights[0] + "\n\n" + content


                    search_result = SearchResult(
                            query=query,
                            url=result.url,
                            title=result.title,
                            content=content,
                            score=getattr(result, 'score', 0.8)
                            )
                    all_results.append(search_result)

                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"❌ Exa search failed for '{query}': {e}")
        print(f"📄 Found {len(all_results)} sources")
        return sorted(all_results, key=lambda x: x.score, reverse=True)[:5]


class SummarizerAgent:
    def __init__(self, llama_client:LlamaCppClient):
        self.client = llama_client

    async def summarize_sources(self, search_results: List[SearchResult]) -> List[SourceSummary]:
        summaries = []
        for result in search_results:
            summary_prompt = f"""Analyze this source and extract key information about the research topic.

            Source: {result.title}
            URL: {result.url}
            Quality Score: {result.score}

            Content:
            {result.content}

            Extract:
            - key_points: List of 3-6 main insights (be specific and factual)
            - confidence: How confident are you in this summary? (0.0-1.0)
            - content_quality: Assess as 'high', 'medium', or 'low' based on depth and and credibility

            Provide structured analysis focusing on actionable insights and concrete findings."""

            try:
                class SummaryResponse(BaseModel):
                    key_points: List[str] = Field(min_items=2, max_items=8)
                    confidence: float=Field(ge=0.0, le=1.0)
                    content_quality: str = Field(pattern="^(high|medium|low)$")

                summary_data = await self.client.generate_structured(
                    summary_prompt, SummaryResponse, max_retries=2
                    )

                summary = SourceSummary(
                    source=result.title,
                    url=result.url,
                    key_points=summary_data.key_points,
                    confidence=summary_data.confidence,
                    content_quality=summary_data.content_quality
                    )

                summaries.append(summary)

            except Exception as e:
                print(f"⚠️  Failed to get structured summary for {result.title}: {e}")
                basic_summary =await self.client.generate(summary_prompt, max_tokens=300)
                summary =  SourceSummary(
                    source=result.title,
                    url=result.url,
                    key_points=[basic_summary],
                    confidence=0.5, 
                    content_quality='medium')
                summaries.append(summary)
        return summaries

class CriticAgent:
    def __init__(self, llama_client: LlamaCppClient):
        self.client = llama_client

    async def critique_research(self, topic: str, summaries: List[SourceSummary]) -> ResearchCritique:

        summaries_text = "\n\n".join([
            f"Source: {s.source} (Quality: {s.content_quality}, Confidence: {s.confidence})\n"
            f"Key Points: {'; '.join(s.key_points)}"
            for s in summaries
            ])
        critique_prompt = f"""You are an expert research critic analyzing research on '{topic}'.

        Sources Analyzed:
        {summaries_text}

        Provide a structured critical analysis with specific scores and lists:

        - overall_quality: Rate overall research quality (0-1)
        - source_credibility: Rate source credibility (0-1)
        - coverage_gaps: List specific missing topics/perspectives (be specific)
        - potential_biases: List potential biases or limitations found
        - consistency_issues: List any contradictions between sources
        - recommendations: List specific actions to improve this research 

        Focus on being constructive and specific in your feedback."""

        try:
            critique = await self.client.generate_structured(
                critique_prompt, ResearchCritique, max_retries=2
                )
            print(f"🔍 Generated structured critique (Quality: {critique.overall_quality})")
            return critique

        except Exception as e:
            print(f"⚠️  Failed to get structured critique: {e}")

            return ResearchCritique(
                overall_quality=0.5,
                source_credibility=0.5,
                coverage_gaps=["Unable to assess due to parsing error"],
                potential_biases=["Analysis incomplete"],
                consistency_issues=[],
                recommendations=["Retry critique with better prompt"]
            )

class ReportGenerator:
    def __init__(self, llama_client:LlamaCppClient):
        self.client = llama_client

    async def generate_final_report(self, topic:str, summaries: List[SourceSummary],
        critique: ResearchCritique) ->ResearchReport:

        all_points = []
        for summary in summaries:
            all_points.extend(summary.key_points)
        report_prompt = f"""Create a comprehensive research report on '{topic}'.

        Key Findings from {len(summaries)} Sources:
        {'; '.join(all_points[:15])}

        Critical Assessment:
        - Overall Quality: {critique.overall_quality}
        - Source Credibility: {critique.source_credibility}
        - Gaps: {'; '.join(critique.coverage_gaps[:3])}
        - Recommendations: {'; '.join(critique.recommendations[:3])}

        Generate a structured report with:
        - executive_summary: 2-3 sentence overview of key findings
        - main_findings: List of 4-6 most important discoveries
        - implications: List of 3-5 practical applications or consequences
        - knowledge_gaps: List of 3-5 areas needing more research
        - conclusions: List of 3-4 clear takeaways
        - research_quality_score: Overall score (0-10) based on source quality and coverage

        Make it professional, actionable, and evidence-based."""

        try:
            report = await self.client.generate_structured(
                report_prompt, ResearchReport, max_retries=2
                )

            report.topic = topic
            report.timestamp = datetime.now()
            report.sources_analyzed = len(summaries)

            print(f"📊 Generated structured report (Quality: {report.research_quality_score}")
            return report
        except Exception as e:
            print(f"⚠️  Failed to get structured report: {e}")
            # Fallback report
            return ResearchReport(
                topic=topic,
                timestamp=datetime.now(),
                executive_summary=f"Research conducted on {topic} with {len(summaries)} sources analyzed.",
                main_findings=["Unable to generate structured findings due to parsing error"],
                implications=["Report generation needs improvement"],
                knowledge_gaps=["Structured output validation failed"],
                conclusions=["System needs debugging"],
                sources_analyzed=len(summaries),
                research_quality_score=critique.overall_quality*10
            )

class ResearchCoordinator:
    def __init__(self, exa_api_key: str):
        self.web_searcher = WebSearchAgent(
            LlamaCppClient("http://localhost:8001", "qwen2.5-0.5b-instruct"),
            exa_api_key
        )
        self.summarizer = SummarizerAgent(
            LlamaCppClient("http://localhost:8002", "qwen2.5-0.5b-instruct")  
        )
        self.critic = CriticAgent(
            LlamaCppClient("http://localhost:8003", "qwen2.5-1.5b-instruct")
        )
        self.report_generator = ReportGenerator(
            LlamaCppClient("http://localhost:8003", "qwen2.5-1.5b-instruct")  # Use critic model
        )

    async def health_check(self) -> bool:
        services = [
            ("Web Searcher", "http://localhost:8001"),
            ("Summarizer", "http://localhost:8002"), 
            ("Critic", "http://localhost:8003")
        ]
        print("🏥 Health checking services...")
        all_healthy = True 

        for name, url in services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            print(f"✅ {name} is healthy")
                        else:
                            print(f"❌ {name} returned {resp.status}")
                            all_healthy=False
            except Exception as e:
                print(f"❌ {name} is down: {e}")
                all_healthy=False
        return all_healthy

    async def research_topic(self, topic:str) -> CompleteResearch:
        start_time = time.time()

        print(f"\n🚀 Starting research on: {topic}")
        try:
            print("\n📡 Phase 1: Web Search (Exa API)")
            search_results = await self.web_searcher.search_web(topic)

            if not search_results:
                raise Exception('No search Results Found!!!')

            print("\n📋 Phase 2: Structured Summarization")
            summaries = await self.summarizer.summarize_sources(search_results)
            

            print("\n🔍 Phase 3: Structured Critical Analysis")
            critique = await self.critic.critique_research(topic, summaries)

            print("\n📊 Phase 4: Structured Report Generation")
            final_report = await self.report_generator.generate_final_report(topic, summaries, critique)


            complete_research = CompleteResearch(
                topic=topic,
                search_results=search_results,
                summaries=summaries,
                critique=critique,
                final_report=final_report,
                timestamp=datetime.now()
                )
            elapsed = time.time() -start_time
            print(f"\n✅ Structured research completed in {elapsed:.2f} seconds")

            return complete_research
        except Exception as e:
            print(f"❌ Research pipeline failed: {e}")
            raise


    def display_result(self, research:CompleteResearch):
        print("\n" + "="*80)
        print(f"RESEARCH REPORT: {research.topic.upper()}")
        print("="*80)
        
       
        print(f"\n📋 EXECUTIVE SUMMARY:")
        print("-" * 40)
        print(research.final_report.executive_summary)
        
        
        print(f"\n🔍 MAIN FINDINGS:")
        print("-" * 40)
        for i, finding in enumerate(research.final_report.main_findings, 1):
            print(f"{i}. {finding}")
        
        
        print(f"\n📊 RESEARCH QUALITY METRICS:")
        print("-" * 40)
        print(f"Overall Quality: {research.critique.overall_quality*10:.2f}/10.0")
        print(f"Source Credibility: {research.critique.source_credibility*10:.2f}/10.0")
        print(f"Final Report Score: {research.final_report.research_quality_score:.2f}/10.0")
        print(f"Sources Analyzed: {research.final_report.sources_analyzed}")
        
        
        print(f"\n⚠️  CRITICAL ANALYSIS:")
        print("-" * 40)
        if research.critique.coverage_gaps:
            print("Coverage Gaps:")
            for gap in research.critique.coverage_gaps[:3]:
                print(f"  • {gap}")
        
        if research.critique.potential_biases:
            print("Potential Biases:")
            for bias in research.critique.potential_biases[:3]:
                print(f"  • {bias}")
        
        #
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(research.critique.recommendations[:3], 1):
            print(f"{i}. {rec}")
        
        
        print(f"\n📚 SOURCES ANALYZED:")
        print("-" * 40)
        for i, summary in enumerate(research.summaries, 1):
            print(f"{i}. {summary.source} ({summary.content_quality} quality)")
            print(f"   URL: {summary.url}")
            print(f"   Confidence: {summary.confidence:.2f}")

    async def save_research(self, research: CompleteResearch, filename:str = None):
        if not filename:
            safe_topic = ''.join(c for c in research.topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            timestamp = research.timestamp.strftime('%Y%m%d_%H%M%S')
            filename = f"research_{safe_topic}_{timestamp}"

        json_file = f'{filename}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(research.model_dump(default=str), f, indent=2, ensure_ascii=False)

        md_file = f"{filename}.md"
        md_content = f"""# Research Report: {research.topic}

**Generated:** {research.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Quality Score:** {research.final_report.research_quality_score}/10
**Sources:** {research.final_report.sources_analyzed}

## Executive Summary
{research.final_report.executive_summary}

## Main Findings
"""
        
        for i, finding in enumerate(research.final_report.main_findings, 1):
            md_content += f"{i}. {finding}\n"
        
        md_content += f"""
## Implications & Applications
"""
        for i, implication in enumerate(research.final_report.implications, 1):
            md_content += f"{i}. {implication}\n"
        
        md_content += f"""
## Knowledge Gaps
"""
        for i, gap in enumerate(research.final_report.knowledge_gaps, 1):
            md_content += f"{i}. {gap}\n"
        
        md_content += f"""
## Critical Analysis

**Overall Quality:** {research.critique.overall_quality*10:.2f}/10

**Source Credibility:** {research.critique.source_credibility*10:.2f}/10

### Coverage Gaps
"""
        for gap in research.critique.coverage_gaps:
            md_content += f"- {gap}\n"
        
        md_content += f"""
### Recommendations
"""
        for rec in research.critique.recommendations:
            md_content += f"- {rec}\n"
        
        md_content += f"""
## Sources
"""
        for i, summary in enumerate(research.summaries, 1):
            md_content += f"""
### {i}. {summary.source}
**URL:** {summary.url}
**Quality:** {summary.content_quality} | **Confidence:** {summary.confidence:.2f}

**Key Points:**
"""
            for point in summary.key_points:
                md_content += f"- {point}\n"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)


async def main(debug: bool=True):
    exa_api_key = os.getenv('EXA_API_KEY')
    if not exa_api_key:
        print("❌ EXA_API_KEY environment variable not set!")
        print("Please set your Exa API key:")
        print("export EXA_API_KEY='your-api-key-here'")
        return

    # web_searcher = WebSearchAgent(
    #         LlamaCppClient("http://localhost:8001", "qwen2.5-0.5b-instruct"),
    #         exa_api_key
    #     )

    # results = await web_searcher.search_web('AI in HealthCare')
    # summarizer = SummarizerAgent(
    #     LlamaCppClient("http://localhost:8002", "qwen2.5-0.5b-instruct")
    #     )
    # summary = await summarizer.summarize_sources(results)
    # print(summary)

    coordinator = ResearchCoordinator(exa_api_key)
    print("🌐 Powered by Exa API + Structured Validation")

    if not await coordinator.health_check():
        print("\n❌ Some llama.cpp services are down!")
        print("\nTo start services, run:")
        print("./llama.cpp/server -m qwen2.5-0.5b-instruct.gguf -p 8001 &")
        print("./llama.cpp/server -m qwen2.5-0.5b-instruct.gguf -p 8002 &") 
        print("./llama.cpp/server -m qwen2.5-1.5b-instruct.gguf -p 8003 &")
        return
    print("\n✅ All services healthy!")
    print("🌐 Exa API configured")
    print("📐 Pydantic validation enabled")
    
    # Interactive topic selection
    print("\n" + "="*60)
    print("STRUCTURED RESEARCH AGENT")
    print("="*60)
    
    print(f"What topic would you like to explore")
    research_topic = input("> ").strip()

    if not research_topic:
        print("❌ No topic provided!")
        if debug:
            print("Come back later when U have a topic ASSHOLE")
        else:
            print('COme back later')
        return
        
    try:
        print(f"\n🎯 Researching: {research_topic}")
        research = await coordinator.research_topic(research_topic)
        
        # Display structured results
        coordinator.display_result(research)
        
        # Ask if user wants to save
        print(f"\n💾 Save structured research data? (y/n): ", end="")
        save_choice = input().strip().lower()
        if save_choice in ['y', 'yes']:
            await coordinator.save_research(research)
            
    except Exception as e:
        print(f"❌ Research failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
